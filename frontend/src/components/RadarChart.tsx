type RadarAxis = {
  label: string;
  value: number;
};

type RadarChartProps = {
  axes: RadarAxis[];
  color?: string;
  size?: number;
};

function RadarChart({ axes, color = "#83C1A7", size = 300 }: RadarChartProps): JSX.Element {
  const cx = size / 2;
  const cy = size / 2;
  const radius = size * 0.30;
  const levels = 5;
  const n = axes.length;

  const getAngle = (i: number): number => -Math.PI / 2 + (i * 2 * Math.PI) / n;

  const getPoint = (i: number, level: number): [number, number] => {
    const a = getAngle(i);
    return [cx + level * radius * Math.cos(a), cy + level * radius * Math.sin(a)];
  };

  const gridCircles = Array.from({ length: levels }, (_, l) => (
    <circle
      key={l}
      cx={cx}
      cy={cy}
      r={((l + 1) / levels) * radius}
      fill="none"
      stroke="rgba(255,255,255,0.1)"
      strokeWidth={1}
    />
  ));

  const gridLines = axes.map((_, i) => {
    const [x, y] = getPoint(i, 1);
    return (
      <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="rgba(255,255,255,0.1)" strokeWidth={1} />
    );
  });

  const dataPoints = axes.map((axis, i) => getPoint(i, Math.max(0.02, axis.value)));
  const polygonPoints = dataPoints.map(([x, y]) => `${x},${y}`).join(" ");

  const labelPad = 26;
  const labels = axes.map((axis, i) => {
    const a = getAngle(i);
    const lx = cx + (radius + labelPad) * Math.cos(a);
    const ly = cy + (radius + labelPad) * Math.sin(a);
    const anchor: "start" | "middle" | "end" =
      lx < cx - 8 ? "end" : lx > cx + 8 ? "start" : "middle";
    return (
      <text
        key={i}
        x={lx}
        y={ly}
        textAnchor={anchor}
        dominantBaseline="middle"
        fill="rgba(255,255,255,0.65)"
        fontSize={11}
        fontFamily='"Google Sans","Open Sans",sans-serif'
      >
        {axis.label}
      </text>
    );
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} overflow="visible">
      {gridCircles}
      {gridLines}
      <polygon
        points={polygonPoints}
        fill={color}
        fillOpacity={0.25}
        stroke={color}
        strokeWidth={2}
        strokeLinejoin="round"
      />
      {dataPoints.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={4} fill={color} />
      ))}
      {labels}
    </svg>
  );
}

export default RadarChart;
