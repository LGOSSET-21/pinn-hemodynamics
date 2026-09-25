const FlowMath = (() => {
  function locate(axis, value) {
    const v = Math.max(axis[0], Math.min(axis[axis.length - 1], value));
    let i = 0;
    while (i < axis.length - 2 && axis[i + 1] < v) i++;
    return [i, (v - axis[i]) / (axis[i + 1] - axis[i])];
  }
  function linear(axis, values, value) {
    const [i, a] = locate(axis, value);
    return values[i] * (1 - a) + values[i + 1] * a;
  }
  function sample(values, rs, ts, r, t) {
    const [i, a] = locate(ts, t);
    return linear(rs, values[i], r) * (1 - a) + linear(rs, values[i + 1], r) * a;
  }
  function cumulative(values, ts) {
    const out = [values[0].map(() => 0)];
    for (let i = 1; i < ts.length; i++) {
      out.push(values[i].map((v, j) => out[i - 1][j] +
        (ts[i] - ts[i - 1]) * (v + values[i - 1][j]) / 2));
    }
    return out;
  }
  function integral(values, acc, rs, ts, r, t) {
    const [i] = locate(ts, t);
    const dt = Math.max(0, Math.min(ts[ts.length - 1], t) - ts[i]);
    return linear(rs, acc[i], r) + dt *
      (linear(rs, values[i], r) + sample(values, rs, ts, r, t)) / 2;
  }
  return {linear, sample, cumulative, integral};
})();
