USE bike;

# Analysis1:  which brand is popular within each category
WITH CTE AS
(
	SELECT pro.product_id, pro.product_name, pro.brand_id, pro.category_id, cat.category_name, brands.brand_name
	FROM products AS pro
	JOIN categories AS cat
		ON pro.category_id = cat.category_id
	JOIN brands
		ON pro.brand_id = brands.brand_id
),
agg AS
(
	SELECT category_name, brand_name, COUNT(*) AS product_count
    FROM CTE
    GROUP BY category_name, brand_name
)
SELECT *,
ROW_NUMBER() OVER(PARTITION BY category_name ORDER BY product_count DESC) AS `rank`
FROM agg
;



# Analysis2: check the total sales in each store
WITH CTE AS
(
	SELECT stores.store_name, order_items.quantity, order_items.list_price, order_items.discount
    FROM stores 
    JOIN orders
		ON stores.store_id = orders.store_id
    JOIN order_items 
		ON orders.order_id = order_items.order_id
)
SELECT store_name, ROUND(SUM((quantity * (1-discount)*list_price)),3) AS total_sales
FROM CTE
GROUP BY store_name
ORDER BY total_sales DESC;


# Analysis3: which category bike was sold the most in each store
WITH CTE AS
(
	SELECT
		orders.store_id,
		categories.category_name,
		order_items.quantity
	FROM orders
	JOIN order_items
		ON orders.order_id = order_items.order_id
	JOIN products
		ON order_items.product_id = products.product_id
	JOIN categories
		ON products.category_id = categories.category_id
),
agg AS
(
	SELECT store_id, category_name, SUM(quantity) AS total_quantity
	FROM CTE
	GROUP BY store_id, category_name
)
SELECT *,
ROW_NUMBER() OVER(PARTITION BY store_id ORDER BY total_quantity DESC) AS rank_num
FROM agg;


# Analysis4: can all orders ship before the required date?
SELECT
	store_id,
	COUNT(*) AS total_orders,
	SUM(CASE 
		WHEN shipped_date > required_date THEN 1 
		ELSE 0 
	END) AS delayed_orders,
	ROUND(
		SUM(CASE WHEN shipped_date > required_date THEN 1 ELSE 0 END)
		/ COUNT(*),
		3
	) AS delay_rate
FROM orders
GROUP BY store_id;


SELECT *
FROM stores;