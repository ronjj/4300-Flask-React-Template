import { useEffect, useRef } from "react";

const GRID_SCALE = 6;
const HEAT_PER_MOVE = 0.04;
const SPREAD_RADIUS = 22;
const DECAY = 0.991;

type RGBA = [number, number, number, number];

const STOPS: Array<[number, ...RGBA]> = [
  [0.00,  0,   0,  0, 0.00],
  [0.10,  0,  90, 40, 0.14],
  [0.35, 15, 150, 60, 0.34],
  [0.70, 30, 190, 80, 0.50],
  [1.00, 55, 215, 95, 0.62],
];

function heatColor(v: number): RGBA {
  for (let i = 1; i < STOPS.length; i++) {
    const [t0, r0, g0, b0, a0] = STOPS[i - 1];
    const [t1, r1, g1, b1, a1] = STOPS[i];
    if (v <= t1) {
      const t = (v - t0) / (t1 - t0);
      return [
        r0 + (r1 - r0) * t,
        g0 + (g1 - g0) * t,
        b0 + (b1 - b0) * t,
        a0 + (a1 - a0) * t,
      ];
    }
  }
  return [55, 215, 95, 0.62];
}

function WelcomeHeatmap(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const parent = canvas?.parentElement;
    if (!canvas || !parent) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let gridW = 0;
    let gridH = 0;
    let heat: Float32Array = new Float32Array(0);

    const offscreen = document.createElement("canvas");
    const offCtx = offscreen.getContext("2d")!;

    const resize = () => {
      canvas.width  = parent.offsetWidth;
      canvas.height = parent.offsetHeight;
      gridW = Math.ceil(canvas.width  / GRID_SCALE);
      gridH = Math.ceil(canvas.height / GRID_SCALE);
      heat = new Float32Array(gridW * gridH);
      offscreen.width  = gridW;
      offscreen.height = gridH;
    };
    resize();

    const ro = new ResizeObserver(resize);
    ro.observe(parent);

    const kernelR = SPREAD_RADIUS;
    const sigma   = kernelR / 2.5;
    const kernel: Float32Array = new Float32Array((2 * kernelR + 1) * (2 * kernelR + 1));
    for (let dy = -kernelR; dy <= kernelR; dy++) {
      for (let dx = -kernelR; dx <= kernelR; dx++) {
        const dist2 = dx * dx + dy * dy;
        if (dist2 > kernelR * kernelR) continue;
        kernel[(dy + kernelR) * (2 * kernelR + 1) + (dx + kernelR)] =
          Math.exp(-dist2 / (2 * sigma * sigma));
      }
    }

    let rafId: number;

    const onMove = (e: MouseEvent) => {
      const { left, top } = parent.getBoundingClientRect();
      const mx = (e.clientX - left) / GRID_SCALE;
      const my = (e.clientY - top)  / GRID_SCALE;

      const x0 = Math.max(0,        Math.floor(mx) - kernelR);
      const x1 = Math.min(gridW - 1, Math.floor(mx) + kernelR);
      const y0 = Math.max(0,        Math.floor(my) - kernelR);
      const y1 = Math.min(gridH - 1, Math.floor(my) + kernelR);

      for (let gy = y0; gy <= y1; gy++) {
        const dy = Math.floor(gy - my) + kernelR;
        for (let gx = x0; gx <= x1; gx++) {
          const dx = Math.floor(gx - mx) + kernelR;
          const k  = kernel[dy * (2 * kernelR + 1) + dx];
          if (k === 0) continue;
          const idx = gy * gridW + gx;
          heat[idx] = Math.min(1, heat[idx] + HEAT_PER_MOVE * k);
        }
      }
    };

    const tick = () => {
      for (let i = 0; i < heat.length; i++) heat[i] *= DECAY;

      const img  = offCtx.createImageData(gridW, gridH);
      const data = img.data;
      for (let i = 0; i < heat.length; i++) {
        const v = heat[i];
        if (v < 0.005) { data[i * 4 + 3] = 0; continue; }
        const [r, g, b, a] = heatColor(v);
        data[i * 4 + 0] = Math.round(r);
        data[i * 4 + 1] = Math.round(g);
        data[i * 4 + 2] = Math.round(b);
        data[i * 4 + 3] = Math.round(a * 255);
      }
      offCtx.putImageData(img, 0, 0);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);

      rafId = requestAnimationFrame(tick);
    };

    parent.addEventListener("mousemove", onMove);
    rafId = requestAnimationFrame(tick);

    return () => {
      parent.removeEventListener("mousemove", onMove);
      cancelAnimationFrame(rafId);
      ro.disconnect();
    };
  }, []);

  return <canvas ref={canvasRef} className="welcome-heatmap" />;
}

export default WelcomeHeatmap;

