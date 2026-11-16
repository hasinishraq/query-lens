SELECT tier FROM seller_tier_history WHERE seller_id IN (SELECT seller_id FROM sellers WHERE name = 'Meghna Mart') AND DATE '2025-07-01' >= valid_from AND DATE '2025-07-01' < valid_to
