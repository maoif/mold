#include "mold.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>

namespace mold::elf {

using E = LARCH64;

static void write_plt_header(Context<E> &ctx) {
  u32 *buf = (u32 *)(ctx.buf + ctx.plt->shdr.sh_offset);

  static const u32 plt0[] = {
    0x00000397, // auipc  t2, %pcrel_hi(.got.plt)
    0x41c30333, // sub    t1, t1, t3               # .plt entry + hdr + 12
    0x0003be03, // ld     t3, %pcrel_lo(1b)(t2)    # _dl_runtime_resolve
    0xfd430313, // addi   t1, t1, -44              # .plt entry
    0x00038293, // addi   t0, t2, %pcrel_lo(1b)    # &.got.plt
    0x00135313, // srli   t1, t1, 1                # .plt entry offset
    0x0082b283, // ld     t0, 8(t0)                # link map
    0x000e0067, // jr     t3
  };

  u64 gotplt = ctx.gotplt->shdr.sh_addr;
  u64 plt = ctx.plt->shdr.sh_addr;

  memcpy(buf, plt0, sizeof(plt0));
  write_utype(buf, gotplt - plt);
  write_itype(buf + 2, gotplt - plt);
  write_itype(buf + 4, gotplt - plt);
}

static void write_plt_entry(Context<E> &ctx, Symbol<E> &sym) {
  u32 *ent = (u32 *)(ctx.buf + ctx.plt->shdr.sh_offset + ctx.plt_hdr_size +
                     sym.get_plt_idx(ctx) * ctx.plt_size);

  static const u32 data[] = {
    0x00000e17, // auipc   t3, %pcrel_hi(function@.got.plt)
    0x000e3e03, // ld      t3, %pcrel_lo(1b)(t3)
    0x000e0367, // jalr    t1, t3
    0x00000013, // nop
  };

  u64 gotplt = sym.get_gotplt_addr(ctx);
  u64 plt = sym.get_plt_addr(ctx);

  memcpy(ent, data, sizeof(data));
  write_utype(ent, gotplt - plt);
  write_itype(ent + 1, gotplt - plt);
}

template <>
void PltSection<E>::copy_buf(Context<E> &ctx) {
  write_plt_header(ctx);
  for (Symbol<E> *sym : symbols)
    write_plt_entry(ctx, *sym);
}

template <>
void PltGotSection<E>::copy_buf(Context<E> &ctx) {
  u32 *buf = (u32 *)(ctx.buf + this->shdr.sh_offset);

  static const u32 data[] = {
    0x00000e17, // auipc   t3, %pcrel_hi(function@.got.plt)
    0x000e3e03, // ld      t3, %pcrel_lo(1b)(t3)
    0x000e0367, // jalr    t1, t3
    0x00000013, // nop
  };

  for (Symbol<E> *sym : symbols) {
    u32 *ent = buf + sym->get_pltgot_idx(ctx) * 4;
    u64 got = sym->get_got_addr(ctx);
    u64 plt = sym->get_plt_addr(ctx);

    memcpy(ent, data, sizeof(data));
    write_utype(ent, got - plt);
    write_itype(ent + 1, got - plt);
  }
}

template <>
void EhFrameSection<E>::apply_reloc(Context<E> &ctx, ElfRel<E> &rel,
                                    u64 offset, u64 val) {
  u8 *loc = ctx.buf + this->shdr.sh_offset + offset;

  switch (rel.r_type) {
  case R_LARCH_ADD32:
    *(u32 *)loc += val;
    return;
  case R_LARCH_SUB8:
    *loc -= val;
    return;
  case R_LARCH_SUB16:
    *(u16 *)loc -= val;
    return;
  case R_LARCH_SUB32:
    *(u32 *)loc -= val;
    return;
  case R_LARCH_SUB6:
    *loc = (*loc & 0b1100'0000) | ((*loc - val) & 0b0011'1111);
    return;
  case R_LARCH_SET6:
    *loc = (*loc & 0b1100'0000) | (val & 0b0011'1111);
    return;
  case R_LARCH_SET8:
    *loc = val;
    return;
  case R_LARCH_SET16:
    *(u16 *)loc = val;
    return;
  case R_LARCH_32_PCREL:
    *(u32 *)loc = val - this->shdr.sh_addr - offset;
    return;
  }
  Fatal(ctx) << "unsupported relocation in .eh_frame: " << rel;
}

template <>
void InputSection<E>::apply_reloc_alloc(Context<E> &ctx, u8 *base) {
  ElfRel<E> *dynrel = nullptr;
  std::span<ElfRel<E>> rels = get_rels(ctx);
  i64 frag_idx = 0;

  if (ctx.reldyn)
    dynrel = (ElfRel<E> *)(ctx.buf + ctx.reldyn->shdr.sh_offset +
                           file.reldyn_offset + this->reldyn_offset);

  for (i64 i = 0; i < rels.size(); i++) {
    const ElfRel<E> &rel = rels[i];
    if (rel.r_type == R_LARCH_NONE || rel.r_type == R_LARCH_GNU_VTINHERIT ||
        rel.r_type == R_LARCH_GNU_VTENTRY)
      continue;

    Symbol<E> &sym = *file.symbols[rel.r_sym];
    i64 r_offset = rel.r_offset + r_deltas[i];
    u8 *loc = base + r_offset;

    const SectionFragmentRef<E> *frag_ref = nullptr;
    if (rel_fragments && rel_fragments[frag_idx].idx == i)
      frag_ref = &rel_fragments[frag_idx++];

#define S   (frag_ref ? frag_ref->frag->get_addr(ctx) : sym.get_addr(ctx))
#define A   (frag_ref ? frag_ref->addend : rel.r_addend)
#define P   (output_section->shdr.sh_addr + offset + r_offset)
#define G   (sym.get_got_addr(ctx) - ctx.got->shdr.sh_addr)
#define GOT ctx.got->shdr.sh_addr

    auto overflow_check = [&](i64 val, i64 lo, i64 hi) {
            if (val < lo || hi <= val)
                    Error(ctx) << *this << ": relocation " << rel << " against "
                               << sym << " out of range: " << val << " is not in ["
                               << lo << ", " << hi << ")";
    };

    auto write8 = [&](u64 val) {
            overflow_check(val, 0, 1 << 8);
            *loc = val;
    };

    auto write8s = [&](u64 val) {
            overflow_check(val, -(1 << 7), 1 << 7);
            *loc = val;
    };

    auto write16 = [&](u64 val) {
            overflow_check(val, 0, 1 << 16);
            *(u16 *)loc = val;
    };

    auto write16s = [&](u64 val) {
            overflow_check(val, -(1 << 15), 1 << 15);
            *(u16 *)loc = val;
    };

    auto write32 = [&](u64 val) {
            overflow_check(val, 0, (i64)1 << 32);
            *(u32 *)loc = val;
    };

    auto write32s = [&](u64 val) {
            overflow_check(val, -((i64)1 << 31), (i64)1 << 31);
            *(u32 *)loc = val;
    };

    auto write64 = [&](u64 val) {
            *(u64 *)loc = val;
    };

    switch (rel.r_type) {
    case R_LARCH_32:
      *(u32 *)loc = S + A;
      break;
    case R_LARCH_64:
      if (sym.is_absolute() || !ctx.arg.pic) {
        *(u64 *)loc = S + A;
      } else if (sym.is_imported) {
        *dynrel++ = {P, R_LARCH_64, (u32)sym.get_dynsym_idx(ctx), A};
        *(u64 *)loc = A;
      } else {
        if (!is_relr_reloc(ctx, rel))
          *dynrel++ = {P, R_LARCH_RELATIVE, 0, (i64)(S + A)};
        *(u64 *)loc = S + A;
      }
      break;
    case R_LARCH_TLS_DTPMOD32:
    case R_LARCH_TLS_DTPMOD64:
    case R_LARCH_TLS_DTPREL32:
    case R_LARCH_TLS_DTPREL64:
    case R_LARCH_TLS_TPREL32:
    case R_LARCH_TLS_TPREL64:
      Error(ctx) << *this << ": unsupported relocation: " << rel;
      break;
    case R_LARCH_BRANCH:
      write_btype((u32 *)loc, S + A - P);
      break;
    case R_LARCH_JAL:
      write_jtype((u32 *)loc, S + A - P);
      break;
    case R_LARCH_CALL:
    case R_LARCH_CALL_PLT: {
      if (r_deltas[i + 1] - r_deltas[i] != 0) {
        // auipc + jalr -> jal
        assert(r_deltas[i + 1] - r_deltas[i] == -4);
        u32 jalr = *(u32 *)&contents[rels[i].r_offset + 4];
        *(u32 *)loc = (0b11111'000000 & jalr) | 0b101111;
        write_jtype((u32 *)loc, S + A - P);
      } else {
        u64 val = sym.esym().is_undef_weak() ? 0 : S + A - P;
        write_utype((u32 *)loc, val);
        write_itype((u32 *)(loc + 4), val);
      }
      break;
    }
    case R_LARCH_GOT_HI20:
      *(u32 *)loc = G + GOT + A - P;
      break;
    case R_LARCH_TLS_GOT_HI20:
      *(u32 *)loc = sym.get_gottp_addr(ctx) + A - P;
      break;
    case R_LARCH_TLS_GD_HI20:
      *(u32 *)loc = sym.get_tlsgd_addr(ctx) + A - P;
      break;
    case R_LARCH_PCREL_HI20:
      if (sym.esym().is_undef_weak()) {
        // Calling an undefined weak symbol does not make sense.
        // We make such call into an infinite loop. This should
        // help debugging of a faulty program.
        *(u32 *)loc = P;
      } else {
        *(u32 *)loc = S + A - P;
      }
      break;
    case R_LARCH_PCREL_LO12_I:
      assert(sym.get_input_section() == this);
      assert(sym.value < r_offset);
      write_itype((u32 *)loc, *(u32 *)(base + sym.value));
      break;
    case R_LARCH_LO12_I:
    case R_LARCH_TPREL_LO12_I:
      write_itype((u32 *)loc, S + A);
      break;
    case R_LARCH_PCREL_LO12_S:
      assert(sym.get_input_section() == this);
      assert(sym.value < r_offset);
      write_stype((u32 *)loc, *(u32 *)(base + sym.value));
      break;
    case R_LARCH_LO12_S:
    case R_LARCH_TPREL_LO12_S:
      write_stype((u32 *)loc, S + A);
      break;
    case R_LARCH_HI20:
      write_utype((u32 *)loc, S + A);
      break;
    case R_LARCH_TPREL_HI20:
      write_utype((u32 *)loc, S + A - ctx.tls_begin);
      break;
    case R_LARCH_TPREL_ADD:
      break;
    case R_LARCH_ADD8:
      loc += S + A;
      break;
    case R_LARCH_ADD16:
      *(u16 *)loc += S + A;
      break;
    case R_LARCH_ADD32:
      *(u32 *)loc += S + A;
      break;
    case R_LARCH_ADD64:
      *(u64 *)loc += S + A;
      break;
    case R_LARCH_SUB8:
      loc -= S + A;
      break;
    case R_LARCH_SUB16:
      *(u16 *)loc -= S + A;
      break;
    case R_LARCH_SUB32:
      *(u32 *)loc -= S + A;
      break;
    case R_LARCH_SUB64:
      *(u64 *)loc -= S + A;
      break;
    case R_LARCH_SUB6:
      *loc = (*loc & 0b1100'0000) | ((*loc - (S + A)) & 0b0011'1111);
      break;
    case R_LARCH_SET6:
      *loc = (*loc & 0b1100'0000) | ((S + A) & 0b0011'1111);
      break;
    case R_LARCH_SET8:
      *loc = S + A;
      break;
    case R_LARCH_SET16:
      *(u16 *)loc = S + A;
      break;
    case R_LARCH_SET32:
      *(u32 *)loc = S + A;
      break;
    case R_LARCH_32_PCREL:
      *(u32 *)loc = S + A - P;
      break;
    default:
      Error(ctx) << *this << ": unknown relocation: " << rel;
    }

#undef S
#undef A
#undef P
#undef G
#undef GOT
  }

}

template <>
void InputSection<E>::apply_reloc_nonalloc(Context<E> &ctx, u8 *base) {
  std::span<ElfRel<E>> rels = get_rels(ctx);

  for (i64 i = 0; i < rels.size(); i++) {
    const ElfRel<E> &rel = rels[i];
    if (rel.r_type == R_LARCH_NONE)
      continue;

    Symbol<E> &sym = *file.symbols[rel.r_sym];
    u8 *loc = base + rel.r_offset;

    if (!sym.file) {
      report_undef(ctx, file, sym);
      continue;
    }

    SectionFragment<E> *frag;
    i64 addend;
    std::tie(frag, addend) = get_fragment(ctx, rel);

#define S (frag ? frag->get_addr(ctx) : sym.get_addr(ctx))
#define A (frag ? addend : rel.r_addend)

    auto overflow_check = [&](i64 val, i64 lo, i64 hi) {
            if (val < lo || hi <= val)
                    Error(ctx) << *this << ": relocation " << rel << " against "
                               << sym << " out of range: " << val << " is not in ["
                               << lo << ", " << hi << ")";
    };

    auto write8 = [&](u64 val) {
            overflow_check(val, 0, 1 << 8);
            *loc = val;
    };

    auto write16 = [&](u64 val) {
            overflow_check(val, 0, 1 << 16);
            *(u16 *)loc = val;
    };

    auto write32 = [&](u64 val) {
            overflow_check(val, 0, (i64)1 << 32);
            *(u32 *)loc = val;
    };

    auto write32s = [&](u64 val) {
            overflow_check(val, -((i64)1 << 31), (i64)1 << 31);
            *(u32 *)loc = val;
    };

    switch (rel.r_type) {
    case R_LARCH_32:
      *(u32 *)loc = S + A;
      break;
    case R_LARCH_64:
      if (std::optional<u64> val = get_tombstone(sym))
        *(u64 *)loc = *val;
      else
        *(u64 *)loc = S + A;
      break;
    case R_LARCH_ADD8:
      *loc += S + A;
      break;
    case R_LARCH_ADD16:
      *(u16 *)loc += S + A;
      break;
    case R_LARCH_ADD32:
      *(u32 *)loc += S + A;
      break;
    case R_LARCH_ADD64:
      *(u64 *)loc += S + A;
      break;
    case R_LARCH_SUB8:
      *loc -= S + A;
      break;
    case R_LARCH_SUB16:
      *(u16 *)loc -= S + A;
      break;
    case R_LARCH_SUB32:
      *(u32 *)loc -= S + A;
      break;
    case R_LARCH_SUB64:
      *(u64 *)loc -= S + A;
      break;
    case R_LARCH_SUB6:
      *loc = (*loc & 0b1100'0000) | ((*loc - (S + A)) & 0b0011'1111);
      break;
    case R_LARCH_SET6:
      *loc = (*loc & 0b1100'0000) | ((S + A) & 0b0011'1111);
      break;
    case R_LARCH_SET8:
      *loc = S + A;
      break;
    case R_LARCH_SET16:
      *(u16 *)loc = S + A;
      break;
    case R_LARCH_SET32:
      *(u32 *)loc = S + A;
      break;
    default:
      Fatal(ctx) << *this << ": invalid relocation for non-allocated sections: "
                 << rel;
      break;
    }

#undef S
#undef A
  }
}

template <>
void InputSection<E>::scan_relocations(Context<E> &ctx) {
  assert(shdr().sh_flags & SHF_ALLOC);

  this->reldyn_offset = file.num_dynrel * sizeof(ElfRel<E>);
  std::span<ElfRel<E>> rels = get_rels(ctx);

  // Scan relocations
  for (i64 i = 0; i < rels.size(); i++) {
    const ElfRel<E> &rel = rels[i];
    if (rel.r_type == R_LARCH_NONE)
      continue;

    Symbol<E> &sym = *file.symbols[rel.r_sym];

    if (!sym.file) {
      report_undef(ctx, file, sym);
      continue;
    }

    if (sym.get_type() == STT_GNU_IFUNC) {
      sym.flags |= NEEDS_GOT;
      sym.flags |= NEEDS_PLT;
    }

    switch (rel.r_type) {
    case R_LARCH_32: {
      Action table[][4] = {
        // Absolute  Local    Imported data  Imported code
        {  NONE,     ERROR,   ERROR,         ERROR },      // DSO
        {  NONE,     ERROR,   ERROR,         ERROR },      // PIE
        {  NONE,     NONE,    COPYREL,       CPLT  },      // PDE
      };
      dispatch(ctx, table, i, rel, sym);
      break;
    }
    case R_LARCH_64: {
      Action table[][4] = {
        // Absolute  Local    Imported data  Imported code
        {  NONE,     BASEREL, DYNREL,        DYNREL },     // DSO
        {  NONE,     BASEREL, DYNREL,        DYNREL },     // PIE
        {  NONE,     NONE,    COPYREL,       CPLT   },     // PDE
      };
      dispatch(ctx, table, i, rel, sym);
      break;
    }
    case R_LARCH_TLS_DTPMOD32:
    case R_LARCH_TLS_DTPMOD64:
    case R_LARCH_TLS_DTPREL32:
    case R_LARCH_TLS_DTPREL64:
    case R_LARCH_TLS_TPREL32:
    case R_LARCH_TLS_TPREL64:
      Error(ctx) << *this << ": unsupported relocation: " << rel;
      break;
    case R_LARCH_CALL:
    case R_LARCH_CALL_PLT:
      if (sym.is_imported)
        sym.flags |= NEEDS_PLT;
      break;
    case R_LARCH_GOT_HI20:
      sym.flags |= NEEDS_GOT;
      break;
    case R_LARCH_TLS_GOT_HI20:
      ctx.has_gottp_rel = true;
      sym.flags |= NEEDS_GOTTP;
      break;
    case R_LARCH_TLS_GD_HI20:
      sym.flags |= NEEDS_TLSGD;
      break;
    case R_LARCH_PCREL_HI20:
    case R_LARCH_PCREL_LO12_I:
    case R_LARCH_PCREL_LO12_S:
    case R_LARCH_LO12_I:
    case R_LARCH_LO12_S:
    case R_LARCH_TPREL_HI20:
    case R_LARCH_TPREL_LO12_I:
    case R_LARCH_TPREL_LO12_S:
    case R_LARCH_TPREL_ADD:
    case R_LARCH_ADD8:
    case R_LARCH_ADD16:
    case R_LARCH_ADD32:
    case R_LARCH_ADD64:
    case R_LARCH_SUB8:
    case R_LARCH_SUB16:
    case R_LARCH_SUB32:
    case R_LARCH_SUB64:
    case R_LARCH_ALIGN:
      break;
    case R_LARCH_RVC_BRANCH:
    case R_LARCH_RVC_JUMP:
      break;
    case R_LARCH_RVC_LUI:
      Error(ctx) << *this << ": unsupported relocation: " << rel;
      break;
    case R_LARCH_RELAX:
    case R_LARCH_SUB6:
    case R_LARCH_SET6:
    case R_LARCH_SET8:
    case R_LARCH_SET16:
    case R_LARCH_SET32:
      break;
    case R_LARCH_32_PCREL: {
      Action table[][4] = {
        // Absolute  Local  Imported data  Imported code
        {  ERROR,    NONE,  ERROR,         ERROR },      // DSO
        {  ERROR,    NONE,  COPYREL,       PLT   },      // PIE
        {  NONE,     NONE,  COPYREL,       PLT   },      // PDE
      };
      dispatch(ctx, table, i, rel, sym);
      break;
    }
    default:
      Error(ctx) << *this << ": unknown relocation: " << rel;
    }
  }
}

} // namespace mold::elf
