SELECT CAST(captured_at_utc AS DATE) AS day, COUNT(*) FROM payments WHERE status = 'captured' AND CAST(captured_at_utc AS DATE) BETWEEN DATE '2026-03-01' AND DATE '2026-03-31' GROUP BY 1 ORDER BY 1
