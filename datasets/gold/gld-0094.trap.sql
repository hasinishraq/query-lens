SELECT th.tier FROM seller_tier_history th JOIN sellers s ON s.seller_id = th.seller_id WHERE s.name = 'Meghna Mart' AND th.valid_to = DATE '9999-12-31'
