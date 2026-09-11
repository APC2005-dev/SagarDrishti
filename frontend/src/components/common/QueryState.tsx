import type { UseQueryResult } from '@tanstack/react-query';
import type { ReactNode } from 'react';

import { ApiError } from '../../api/client';

interface Props<T> {
  query: UseQueryResult<T>;
  children: (data: T) => ReactNode;
  empty?: (data: T) => boolean;
  emptyText?: ReactNode;
  notFoundText?: ReactNode;
  compact?: boolean;
}

/** Uniform loading / error / empty handling. Never renders placeholder data as if it were real. */
export function QueryState<T>({ query, children, empty, emptyText, notFoundText, compact }: Props<T>) {
  if (query.isPending) {
    return (
      <div className={`qs qs-loading${compact ? ' compact' : ''}`}>
        <span className="skeleton-bar" />
        <span className="skeleton-bar short" />
      </div>
    );
  }
  if (query.isError) {
    const err = query.error;
    if (err instanceof ApiError && err.status === 404) {
      return <div className={`qs qs-empty${compact ? ' compact' : ''}`}>{notFoundText ?? 'Not available.'}</div>;
    }
    const unavailable = err instanceof ApiError && err.unavailable;
    return (
      <div className={`qs qs-error${compact ? ' compact' : ''}`} role="alert">
        <span className="label">{unavailable ? 'Backend unavailable' : 'Request failed'}</span>
        <span className="muted">{err instanceof ApiError ? err.detail : String(err)}</span>
        {err instanceof ApiError && err.requestId && <span className="mono dim">request {err.requestId}</span>}
        <button className="btn-ghost" onClick={() => query.refetch()}>
          Retry
        </button>
      </div>
    );
  }
  const data = query.data as T;
  if (empty?.(data)) {
    return <div className={`qs qs-empty${compact ? ' compact' : ''}`}>{emptyText ?? 'No data.'}</div>;
  }
  return <>{children(data)}</>;
}
