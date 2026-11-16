SELECT SUM(lt.points) AS value FROM loyalty_transactions lt WHERE lt.txn_type = 'redeem' AND CAST(lt.txn_at AS DATE) BETWEEN DATE '2026-06-01' AND DATE '2026-06-30'
