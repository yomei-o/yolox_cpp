// YOLOX loss on the pure engine: anchor-free decode + IoU loss + BCE(obj/cls),
// with SimOTA (plain, no-grad) providing constant targets.
#pragma once
#include "ops2d.hpp"
#include "simota.hpp"

inline Tensor sq(const Tensor& z) { return mul(z, z); }
// BCE-with-logits (stable), z may be a (constant) tensor: max(x,0) - x*z + log1p(exp(-|x|)).
inline Tensor bce_logits(const Tensor& x, const Tensor& z) {
  return add(sub(clampmin_scalar(x, 0.f), mul(x, z)), op_log1p(op_exp(mul_scalar(op_abs(x), -1.f))));
}

// (1,C,h,w) -> (h*w, C)
inline Tensor chw_to_rows(const Tensor& t) {
  int64_t C = t->shape[1], H = t->shape[2], W = t->shape[3], P = H * W;
  auto o = make_tensor({P, C}, true);
  for (int64_t c = 0; c < C; ++c) for (int64_t p = 0; p < P; ++p) o->data[p*C+c] = t->data[c*P+p];
  o->parents = {t}; Node* op = o.get();
  o->backward_fn = [t, op, C, P] { for (int64_t c=0;c<C;++c) for (int64_t p=0;p<P;++p) t->grad[c*P+p] += op->grad[p*C+c]; };
  return o;
}
inline Tensor concat_rows(const std::vector<Tensor>& xs) {
  int64_t C = last_dim(xs[0]); int64_t R = 0; for (auto& x : xs) R += x->numel()/C;
  auto o = make_tensor({R, C}, true);
  int64_t off = 0;
  for (auto& x : xs) { int64_t r = x->numel()/C; std::copy_n(x->data.data(), r*C, o->data.data()+off*C); off += r; }
  o->parents = xs; Node* op = o.get();
  o->backward_fn = [xs, op, C] { int64_t off=0; for (auto& x: xs){ int64_t r=x->numel()/C; for(int64_t i=0;i<r*C;++i) x->grad[i]+=op->grad[off*C+i]; off+=r; } };
  return o;
}
// IoU of paired cxcywh boxes (n,4)x(n,4) -> (n,1), matching YOLOX bboxes_iou.
inline Tensor iou_rows(const Tensor& p, const Tensor& t) {
  auto px=narrow_col(p,0), py=narrow_col(p,1), pw=narrow_col(p,2), ph=narrow_col(p,3);
  auto tx=narrow_col(t,0), ty=narrow_col(t,1), tw=narrow_col(t,2), th=narrow_col(t,3);
  auto pl=sub(px,mul_scalar(pw,0.5f)), pr=add(px,mul_scalar(pw,0.5f)), pt=sub(py,mul_scalar(ph,0.5f)), pb=add(py,mul_scalar(ph,0.5f));
  auto tl=sub(tx,mul_scalar(tw,0.5f)), tr=add(tx,mul_scalar(tw,0.5f)), tt=sub(ty,mul_scalar(th,0.5f)), tb=add(ty,mul_scalar(th,0.5f));
  auto iw=clampmin_scalar(sub(min2(pr,tr),max2(pl,tl)),0.f);
  auto ih=clampmin_scalar(sub(min2(pb,tb),max2(pt,tt)),0.f);
  auto inter=mul(iw,ih);
  auto uni=add_scalar(sub(add(mul(pw,ph),mul(tw,th)),inter),1e-16f);
  return divi(inter,uni);
}

struct YoloxLoss { Tensor total, iou, obj, cls; };
// raw: per-level (1,85,h,w). anc consts: xs,ys,strides (A). gt for SimOTA. nc.
inline YoloxLoss yolox_loss(std::vector<Tensor> raw, const std::vector<float>& xs,
                            const std::vector<float>& ys, const std::vector<float>& strides,
                            const std::vector<float>& gtb, const std::vector<int64_t>& gtc,
                            int64_t A, int64_t nc, int64_t G) {
  std::vector<Tensor> rr; for (auto& r : raw) rr.push_back(chw_to_rows(r));
  auto raws = concat_rows(rr);                       // (A, 85)
  auto XS = from_data({A,1}, xs), YS = from_data({A,1}, ys), ST = from_data({A,1}, strides);
  // decode (differentiable)
  auto cx = mul(add(narrow_col(raws,0), XS), ST);
  auto cy = mul(add(narrow_col(raws,1), YS), ST);
  auto w  = mul(op_exp(narrow_col(raws,2)), ST);
  auto h  = mul(op_exp(narrow_col(raws,3)), ST);
  auto box = concat_cols({cx, cy, w, h});           // (A,4) cxcywh
  auto obj = narrow_col(raws, 4);                    // (A,1)
  auto cls = narrow_cols(raws, 5, 5+nc);            // (A,nc)

  // SimOTA on detached decoded data
  Assign as = simota_assign(box->data, cls->data, obj->data, xs, ys, strides, gtb, gtc, A, nc, G);
  float nf = (float)std::max<int64_t>(as.num_fg, 1);

  // fg indices + targets (constants)
  std::vector<int64_t> fg;
  std::vector<float> regt, clst(as.num_fg*nc, 0.f), objt(A, 0.f);
  for (int64_t a=0;a<A;++a) if (as.fg[a]>0.5f) { fg.push_back(a); objt[a]=1.f; }
  for (int64_t k=0;k<as.num_fg;++k) {
    int64_t g = as.matched_gt[k];
    for (int j=0;j<4;++j) regt.push_back(gtb[g*4+j]);
    clst[k*nc + as.matched_cls[k]] = as.pred_ious[k];   // onehot * iou (soft, constant)
  }
  auto reg_t = from_data({as.num_fg,4}, regt);
  auto cls_t = from_data({as.num_fg,nc}, clst);
  auto obj_t = from_data({A,1}, objt);

  // loss_iou = sum(1 - iou^2)/nf over fg
  auto box_fg = gather_rows(box, fg);
  auto iou = iou_rows(box_fg, reg_t);
  auto liou = mul_scalar(sum(add_scalar(mul_scalar(sq(iou), -1.f), 1.f)), 1.f/nf);
  // loss_obj = sum BCE(obj_all, fg)/nf
  auto lobj = mul_scalar(sum(bce_logits(obj, obj_t)), 1.f/nf);
  // loss_cls = sum BCE(cls[fg], onehot*iou)/nf
  auto cls_fg = gather_rows(cls, fg);
  auto lcls = mul_scalar(sum(bce_logits(cls_fg, cls_t)), 1.f/nf);

  auto total = add(add(mul_scalar(liou, 5.f), lobj), lcls);
  return {total, liou, lobj, lcls};
}
