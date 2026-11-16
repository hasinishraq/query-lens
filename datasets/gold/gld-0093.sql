SELECT th.tier FROM seller_tier_history th JOIN sellers s ON s.seller_id = th.seller_id WHERE s.name = 'Meghna Mart' AND th.valid_from <= DATE '2025-06-30' AND DATE '2025-06-30' < th.valid_to
