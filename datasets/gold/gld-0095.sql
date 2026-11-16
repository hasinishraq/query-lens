SELECT th.tier FROM seller_tier_history th JOIN sellers s ON s.seller_id = th.seller_id WHERE s.name = 'Meghna Mart' AND th.valid_from <= DATE '2024-12-31' AND DATE '2024-12-31' < th.valid_to
