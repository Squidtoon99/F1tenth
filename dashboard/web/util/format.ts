export function trimNum(value: number, digits = 3): string {
  return value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

export function fmtStat(value: number | null, unit: string): string {
  if (value == null) return "—";
  const text = trimNum(value);
  return unit === "s" ? `${text}s` : text;
}

export function fmtDelta(delta: number | null, unit: string): string {
  if (delta == null) return "";
  const sign = delta > 0 ? "+" : "";
  const suffix = unit === "s" ? "s" : "";
  return `${sign}${trimNum(delta)}${suffix}`;
}

export function fmtMillions(value: number | null): string {
  if (value == null) return "—";
  const millions = value / 1_000_000;
  const text = millions >= 100 ? millions.toFixed(0) : trimNum(millions, 1);
  return `${text}M`;
}

export function fmtBytes(value: number | null): string {
  if (value == null) return "—";
  return `${trimNum(value / (1024 * 1024), 2)} MiB`;
}
