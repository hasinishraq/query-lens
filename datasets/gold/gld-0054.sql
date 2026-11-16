SELECT SUM(sp.gross_bdt - sp.commission_bdt) AS value FROM seller_payouts sp WHERE sp.period_start BETWEEN date_trunc('month', DATE '2024-02-01') AND DATE '2024-02-29'
