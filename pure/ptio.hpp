// Pure-C++ PyTorch .pt (state_dict) reader/writer — no Python, no libtorch. A .pt is a
// ZIP (STORED) holding a pickle (protocol 2) describing an OrderedDict[str->Tensor] plus
// one raw float32 storage per tensor under data/<k>. This handles the state_dict form
// (torch.save(model.state_dict())) — read and write — so weights round-trip C++<->PyTorch
// without a Python bridge. FloatStorage / little-endian only (what our models use).
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>

namespace pt {

struct Tensor { std::string name; std::vector<int64_t> shape; std::vector<float> data; };

// ----------------------------- CRC32 (zip) -----------------------------
inline uint32_t crc32_of(const uint8_t* p, size_t n) {
  static uint32_t T[256]; static bool init = false;
  if (!init) { for (uint32_t i = 0; i < 256; ++i) { uint32_t c = i; for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; T[i] = c; } init = true; }
  uint32_t c = 0xFFFFFFFFu; for (size_t i = 0; i < n; ++i) c = T[(c ^ p[i]) & 0xFF] ^ (c >> 8); return c ^ 0xFFFFFFFFu;
}

// ----------------------------- byte sinks -----------------------------
inline void pu16(std::string& b, uint16_t v) { b.push_back((char)(v & 0xff)); b.push_back((char)(v >> 8)); }
inline void pu32(std::string& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((char)((v >> (8 * i)) & 0xff)); }

// ============================== WRITER ==============================
// pickle helpers
inline void pk_bININT(std::string& p, int64_t v) { p.push_back('J'); pu32(p, (uint32_t)(int32_t)v); }
inline void pk_str(std::string& p, const std::string& s) { p.push_back('X'); pu32(p, (uint32_t)s.size()); p += s; }
inline void pk_global(std::string& p, const std::string& mod, const std::string& name) { p.push_back('c'); p += mod; p.push_back('\n'); p += name; p.push_back('\n'); }

inline std::string build_pickle(const std::vector<Tensor>& ts) {
  std::string p;
  p.push_back((char)0x80); p.push_back(2);                 // PROTO 2
  pk_global(p, "collections", "OrderedDict"); p.push_back(')'); p.push_back('R');   // OrderedDict()
  p.push_back('(');                                        // MARK
  for (size_t i = 0; i < ts.size(); ++i) {
    int64_t numel = 1; for (int64_t d : ts[i].shape) numel *= d;
    pk_str(p, ts[i].name);
    pk_global(p, "torch._utils", "_rebuild_tensor_v2");
    p.push_back('(');                                      // MARK (args)
      p.push_back('(');                                    // MARK (persid tuple)
        pk_str(p, "storage"); pk_global(p, "torch", "FloatStorage"); pk_str(p, std::to_string(i)); pk_str(p, "cpu"); pk_bININT(p, numel);
      p.push_back('t'); p.push_back('Q');                  // TUPLE, BINPERSID
      pk_bININT(p, 0);                                     // storage_offset
      p.push_back('(');                                    // size tuple
        for (int64_t d : ts[i].shape) pk_bININT(p, d);
      p.push_back('t');
      p.push_back('(');                                    // stride tuple (contiguous)
        for (size_t k = 0; k < ts[i].shape.size(); ++k) { int64_t s = 1; for (size_t j = k + 1; j < ts[i].shape.size(); ++j) s *= ts[i].shape[j]; pk_bININT(p, s); }
      p.push_back('t');
      p.push_back((char)0x89);                             // requires_grad = False
      pk_global(p, "collections", "OrderedDict"); p.push_back(')'); p.push_back('R');  // backward hooks {}
    p.push_back('t'); p.push_back('R');                    // TUPLE(args), REDUCE
  }
  p.push_back('u'); p.push_back('.');                      // SETITEMS, STOP
  return p;
}

// zip: one STORED entry, data 64-byte aligned via an extra field.
inline void zip_add(std::string& out, std::string& central, const std::string& name, const std::string& data) {
  uint32_t crc = crc32_of((const uint8_t*)data.data(), data.size());
  uint32_t local_off = (uint32_t)out.size();
  uint32_t hdr = 30 + (uint32_t)name.size();
  uint32_t pad = (64 - (local_off + hdr + 4) % 64) % 64;   // +4 for the extra header
  uint16_t extra = (uint16_t)(4 + pad);
  out += std::string("PK\x03\x04", 4); pu16(out, 20); pu16(out, 0); pu16(out, 0); pu16(out, 0); pu16(out, 0);
  pu32(out, crc); pu32(out, (uint32_t)data.size()); pu32(out, (uint32_t)data.size());
  pu16(out, (uint16_t)name.size()); pu16(out, extra); out += name;
  pu16(out, 0xCAFE); pu16(out, (uint16_t)pad); out.append(pad, '\0');   // extra: id + len + pad
  out += data;
  central += std::string("PK\x01\x02", 4); pu16(central, 20); pu16(central, 20); pu16(central, 0); pu16(central, 0); pu16(central, 0); pu16(central, 0);
  pu32(central, crc); pu32(central, (uint32_t)data.size()); pu32(central, (uint32_t)data.size());
  pu16(central, (uint16_t)name.size()); pu16(central, 0); pu16(central, 0); pu16(central, 0); pu16(central, 0); pu32(central, 0); pu32(central, local_off);
  central += name;
}

inline void save_pt(const std::vector<Tensor>& ts, const std::string& path) {
  std::string out, central; const std::string A = "archive/";
  zip_add(out, central, A + "data.pkl", build_pickle(ts));
  zip_add(out, central, A + "byteorder", "little");
  for (size_t i = 0; i < ts.size(); ++i)
    zip_add(out, central, A + "data/" + std::to_string(i), std::string((const char*)ts[i].data.data(), ts[i].data.size() * 4));
  zip_add(out, central, A + "version", "3\n");
  uint32_t cd_off = (uint32_t)out.size(); out += central;
  out += std::string("PK\x05\x06", 4); pu16(out, 0); pu16(out, 0);
  pu16(out, (uint16_t)ts.size() + 3); pu16(out, (uint16_t)ts.size() + 3);
  pu32(out, (uint32_t)central.size()); pu32(out, cd_off); pu16(out, 0);
  std::ofstream f(path, std::ios::binary); f.write(out.data(), out.size());
}

// ============================== READER ==============================
struct Val {
  enum T { I, S, TUP, GLB, PID, BOOL, DICT, TEN, NONE } t = NONE;
  int64_t i = 0; std::string s; std::vector<Val> tup;
  std::vector<std::pair<Val, Val>> items;                  // DICT
  std::string tkey; std::vector<int64_t> tshape;           // TEN
};

inline std::vector<Tensor> load_pt(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { printf("cannot open %s\n", path.c_str()); std::exit(1); }
  size_t n = f.tellg(); f.seekg(0); std::string buf(n, '\0'); f.read(buf.data(), n);
  const uint8_t* B = (const uint8_t*)buf.data();
  auto u16 = [&](size_t o) { return (uint32_t)B[o] | (B[o + 1] << 8); };
  auto u32 = [&](size_t o) { return (uint32_t)B[o] | (B[o+1]<<8) | (B[o+2]<<16) | ((uint32_t)B[o+3]<<24); };

  // locate End Of Central Directory, walk central dir -> name -> data offset/size
  size_t eocd = n - 22; while (eocd > 0 && u32(eocd) != 0x06054b50) --eocd;
  uint32_t cd = u32(eocd + 16), cnt = u16(eocd + 10);
  std::vector<std::pair<std::string, std::pair<size_t, size_t>>> entries;
  size_t p = cd;
  for (uint32_t e = 0; e < cnt; ++e) {
    uint32_t csize = u32(p + 20), nlen = u16(p + 28), elen = u16(p + 30), clen = u16(p + 32), lo = u32(p + 42);
    std::string nm((const char*)B + p + 46, nlen);
    uint32_t lnlen = u16(lo + 26), lelen = u16(lo + 28);
    entries.push_back({nm, {lo + 30 + lnlen + lelen, csize}});
    p += 46 + nlen + elen + clen;
  }
  auto find = [&](const std::string& suffix) -> std::pair<size_t, size_t> {
    for (auto& e : entries) if (e.first.size() >= suffix.size() && e.first.compare(e.first.size() - suffix.size(), suffix.size(), suffix) == 0) return e.second;
    return {0, 0};
  };

  // ---- pickle VM over data.pkl ----
  auto pk = find("data.pkl"); const uint8_t* P = B + pk.first; size_t PN = pk.second, ip = 0;
  std::vector<Val> st; std::vector<size_t> marks; std::vector<Val> memo(2048);
  auto rd_line = [&]() { std::string s; while (P[ip] != '\n') s.push_back((char)P[ip++]); ++ip; return s; };
  auto rd_u32 = [&]() { uint32_t v = P[ip] | (P[ip+1]<<8) | (P[ip+2]<<16) | ((uint32_t)P[ip+3]<<24); ip += 4; return v; };
  Val result;
  while (ip < PN) {
    uint8_t op = P[ip++];
    if (op == 0x80) { ++ip; }                                          // PROTO
    else if (op == 'c') { Val v; v.t = Val::GLB; std::string m = rd_line(), nm = rd_line(); v.s = m + " " + nm; st.push_back(v); }
    else if (op == 'q') { uint8_t idx = P[ip++]; if (idx >= memo.size()) memo.resize(idx + 1); memo[idx] = st.back(); }   // BINPUT
    else if (op == 'r') { uint32_t idx = rd_u32(); if (idx >= memo.size()) memo.resize(idx + 1); memo[idx] = st.back(); } // LONG_BINPUT
    else if (op == 'h') { uint8_t idx = P[ip++]; st.push_back(memo[idx]); }                                              // BINGET
    else if (op == 'j') { uint32_t idx = rd_u32(); st.push_back(memo[idx]); }                                            // LONG_BINGET
    else if (op == ')') { Val v; v.t = Val::TUP; st.push_back(v); }                                                      // EMPTY_TUPLE
    else if (op == '(') { marks.push_back(st.size()); }                                                                  // MARK
    else if (op == 'X') { uint32_t l = rd_u32(); Val v; v.t = Val::S; v.s.assign((const char*)P + ip, l); ip += l; st.push_back(v); }  // BINUNICODE
    else if (op == 'K') { Val v; v.t = Val::I; v.i = P[ip++]; st.push_back(v); }                                         // BININT1
    else if (op == 'M') { Val v; v.t = Val::I; v.i = P[ip] | (P[ip+1]<<8); ip += 2; st.push_back(v); }                   // BININT2
    else if (op == 'J') { Val v; v.t = Val::I; v.i = (int32_t)rd_u32(); st.push_back(v); }                               // BININT
    else if (op == 0x8a) { uint8_t ln = P[ip++]; int64_t val = 0; for (int k = 0; k < ln; ++k) val |= (int64_t)P[ip++] << (8*k); Val v; v.t = Val::I; v.i = val; st.push_back(v); } // LONG1
    else if (op == 0x88 || op == 0x89) { Val v; v.t = Val::BOOL; v.i = (op == 0x88); st.push_back(v); }                  // NEWTRUE/FALSE
    else if (op == 'N') { st.push_back(Val{}); }                                                                         // NONE
    else if (op == 't') { size_t m = marks.back(); marks.pop_back(); Val v; v.t = Val::TUP; for (size_t k = m; k < st.size(); ++k) v.tup.push_back(st[k]); st.resize(m); st.push_back(v); }  // TUPLE
    else if (op == 0x85 || op == 0x86 || op == 0x87) { int k = op - 0x84; Val v; v.t = Val::TUP; for (int j = 0; j < k; ++j) v.tup.insert(v.tup.begin(), st.back()), st.pop_back(); st.push_back(v); }  // TUPLE1/2/3
    else if (op == 'Q') { Val v; v.t = Val::PID; v.tup = st.back().tup; st.pop_back(); st.push_back(v); }                // BINPERSID (pid tuple -> PID)
    else if (op == 'R') {                                                                                                // REDUCE
      Val args = st.back(); st.pop_back(); Val fn = st.back(); st.pop_back(); Val out;
      if (fn.t == Val::GLB && fn.s == "collections OrderedDict") { out.t = Val::DICT; }
      else if (fn.t == Val::GLB && fn.s == "torch._utils _rebuild_tensor_v2") {
        out.t = Val::TEN; const Val& pid = args.tup[0]; out.tkey = pid.tup[2].s;   // pid: (storage,FloatStorage,key,cpu,numel)
        for (auto& d : args.tup[2].tup) out.tshape.push_back(d.i);                 // size tuple
      } else out.t = Val::NONE;
      st.push_back(out);
    }
    else if (op == 'u') { size_t m = marks.back(); marks.pop_back();       // SETITEMS
      for (size_t k = m; k + 1 < st.size(); k += 2) st[m - 1].items.push_back({st[k], st[k + 1]});
      st.resize(m); }
    else if (op == 's') { Val val = st.back(); st.pop_back(); Val key = st.back(); st.pop_back(); st.back().items.push_back({key, val}); }  // SETITEM
    else if (op == '}') { Val v; v.t = Val::DICT; st.push_back(v); }                                                     // EMPTY_DICT
    else if (op == '.') { result = st.back(); break; }                                                                  // STOP
    else { printf("ptio: unhandled pickle op 0x%02x at %zu\n", op, ip - 1); std::exit(1); }
  }

  std::vector<Tensor> out;
  for (auto& kv : result.items) {
    if (kv.first.t != Val::S || kv.second.t != Val::TEN) continue;
    Tensor t; t.name = kv.first.s; t.shape = kv.second.tshape;
    auto st_ent = find("data/" + kv.second.tkey);
    int64_t numel = 1; for (int64_t d : t.shape) numel *= d;
    t.data.resize(numel); std::memcpy(t.data.data(), B + st_ent.first, numel * 4);
    out.push_back(std::move(t));
  }
  return out;
}

}  // namespace pt
