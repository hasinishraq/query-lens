SELECT SUM(s.shipping_fee_bdt) FROM shipments s WHERE s.shipped_at_utc >= TIMESTAMP '2025-03-01' - INTERVAL 6 HOUR AND s.shipped_at_utc < TIMESTAMP '2025-03-31' + INTERVAL 18 HOUR
