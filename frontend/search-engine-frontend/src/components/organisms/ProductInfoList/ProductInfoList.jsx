import ProductInfo from "../../molecules/ProductInfo/ProductInfo";

function ProductInfoList({ products }) {
  return products && products.length > 0 && (
    <div style={{ padding: "30px" }}>
      {products.map((product, index) => {
        return (
          <ProductInfo 
            key={index}
            title={product.title} 
            imageURL={product.imageURL}
          />
        )
      })}
    </div>
  );
}

export default ProductInfoList;
