import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";

export interface ApiState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
  setData: (d: T | null) => void;
}

/** GET a backend path; path === null means "not yet" (e.g. waiting for a selection). */
export function useApi<T>(path: string | null): ApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(path !== null);
  const [tick, setTick] = useState(0);
  const seq = useRef(0);

  useEffect(() => {
    if (path === null) {
      setLoading(false);
      return;
    }
    const mine = ++seq.current;
    setLoading(true);
    setError(null);
    api<T>(path)
      .then((d) => mine === seq.current && setData(d))
      .catch((e: Error) => mine === seq.current && (setError(e.message), setData(null)))
      .finally(() => mine === seq.current && setLoading(false));
  }, [path, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload, setData };
}
