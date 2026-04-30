# GPT3DS

## Good Petite Transformer with 3 Degrees of Self-attention
Does not stand for anything else

## Building the 3DS app

Install devkitPro with devkitARM and libctru, then build from the `3ds`
directory:

```bash
cd 3ds
export DEVKITPRO=/opt/devkitpro
export DEVKITARM="$DEVKITPRO/devkitARM"
export PATH="$DEVKITARM/bin:$DEVKITPRO/tools/bin:$PATH"
make
```

This produces `3ds/GPT3DS.3dsx` and, when `makerom` is available,
`3ds/GPT3DS.cia`.

For a clean rebuild:

```bash
cd 3ds
make clean
make
```
