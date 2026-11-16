SELECT SUM(gross_bdt) - SUM(commission_bdt) AS v FROM seller_payouts WHERE period_start >= DATE '2025-09-01' AND period_start <= DATE '2025-09-30'
