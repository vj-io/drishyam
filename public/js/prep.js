// Client-side image prep: decode with EXIF orientation, re-encode as JPEG when the file is too big.
export const LIMITS = { bytes: 4_400_000, pixels: 24_000_000 };

const toBlob = (canvas, q) => new Promise((res) => canvas.toBlob(res, 'image/jpeg', q));

/** -> { blob, width, height, note } where blob is safe to upload. */
export async function prepareImage(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  try {
    const { width, height } = bmp;
    if (file.size <= LIMITS.bytes && width * height <= LIMITS.pixels) return { blob: file, width, height, note: '' };
    return await shrink(bmp, file.size);
  } finally {
    bmp.close();
  }
}

async function shrink(bmp, originalBytes) {
  let scale = Math.min(1, Math.sqrt(LIMITS.pixels / (bmp.width * bmp.height)));
  for (;;) {
    const w = Math.round(bmp.width * scale);
    const h = Math.round(bmp.height * scale);
    if (Math.min(w, h) < 64) throw new Error('Could not shrink this image enough to upload.');
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const g = canvas.getContext('2d');
    g.fillStyle = '#fff';
    g.fillRect(0, 0, w, h);
    g.drawImage(bmp, 0, 0, w, h);
    for (let q = 0.92; q >= 0.6 - 1e-9; q -= 0.08) {
      const blob = await toBlob(canvas, q);
      if (blob && blob.size <= LIMITS.bytes) {
        const mb = (n) => (n / 1e6).toFixed(1);
        return { blob, width: w, height: h, note: `shrunk ${mb(originalBytes)} MB to ${mb(blob.size)} MB (JPEG q${q.toFixed(2)}, ${w}×${h})` };
      }
    }
    scale *= 0.85;
  }
}
