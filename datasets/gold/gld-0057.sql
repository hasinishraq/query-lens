SELECT SUM(sp.gross_bdt - sp.commission_bdt) AS value FROM seller_payouts sp WHERE sp.period_start BETWEEN date_trunc('month', DATE '2025-07-01') AND DATE '2025-07-31'
