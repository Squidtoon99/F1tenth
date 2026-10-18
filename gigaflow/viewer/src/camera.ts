export type Point3 = [number, number, number];

export function advanceFollowTarget(
  current: Point3,
  desired: Point3,
  alpha: number,
): { target: Point3; delta: Point3 } {
  const target: Point3 = [
    current[0] + (desired[0] - current[0]) * alpha,
    current[1] + (desired[1] - current[1]) * alpha,
    current[2] + (desired[2] - current[2]) * alpha,
  ];
  return {
    target,
    delta: [
      target[0] - current[0],
      target[1] - current[1],
      target[2] - current[2],
    ],
  };
}
