import React from 'react';

const ProductList = ({ products }) => {
  return (
    <div>
      {products.map((product, index) => (
        <div key={index} className="product">
          <img src={product.image_url} alt={product.name} />
          <a href={product.shopping_url} target="_blank" rel="noopener noreferrer">
            {product.name}
          </a>
        </div>
      ))}
    </div>
  );
};

export default ProductList;
