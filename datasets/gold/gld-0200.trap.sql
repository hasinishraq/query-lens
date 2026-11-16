SELECT CAST(captured_at_utc AS DATE) AS day, COUNT(*) FROM payments WHERE status = 'captured' AND CAST(captured_at_utc AS DATE) BETWEEN DATE '2026-02-01' AND DATE '2026-02-28' GROUP BY 1 ORDER BY 1
