def print_quantiles(series, name, quantiles=None):
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

    s = series.dropna()
    q = s.quantile(quantiles)
    print(f"\n{name} – basic stats")
    print(f"  count: {s.shape[0]}")
    print(f"  min:   {s.min():,.0f}")
    print(f"  max:   {s.max():,.0f}")
    print("  quantiles:")
    for p, val in q.items():
        print(f"    {int(p*100):>2}th: {val:,.0f}")