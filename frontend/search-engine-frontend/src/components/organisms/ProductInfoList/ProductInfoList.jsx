import ProductInfo from "../../molecules/ProductInfo/ProductInfo";
import LineBreak from "../../atoms/LineBreak/LineBreak";

function ProductInfoList({ products }) {
  return products && products.length > 0 && (
    <div 
      style={{ 
        padding: "30px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        width: "70%",
      }}
    >
      {products.map((product, index) => {
        return (
          <>
            <ProductInfo 
              key={index}
              title={product.title} 
              brand={product.brand} 
              price={product.price} 
              description={product.description} 
              imageURL={product.imageURL}
            />
            <LineBreak width='95%' />
          </>
        )
      })}
    </div>
  );
}

export default ProductInfoList;
