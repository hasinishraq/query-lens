SELECT CAST(captured_at_utc AS DATE) AS day, COUNT(*) FROM payments WHERE status = 'captured' AND CAST(captured_at_utc AS DATE) BETWEEN DATE '2025-04-01' AND DATE '2025-04-30' GROUP BY 1 ORDER BY 1
