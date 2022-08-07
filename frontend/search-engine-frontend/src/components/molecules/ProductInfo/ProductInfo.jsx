function ProductInfo({ 
  title,
  brand,
  price,
  description,
  imageURL 
}) {
  let imageThumbnail = "https://image.makewebeasy.net/makeweb/0/GYWrZvZVh/ADVICE/%E0%B8%94%E0%B8%B2%E0%B8%A7%E0%B8%99%E0%B9%8C%E0%B9%82%E0%B8%AB%E0%B8%A5%E0%B8%94_3__1.jpg";
  if (imageURL.length > 0) {
    imageThumbnail = imageURL[0];
  }

  let productDescription = "No description available.";
  if (description.length > 0) {
    productDescription = description[0];
  }

  return (
    <div
      style={{
        display: "flex",
      }}
    >
      <img 
        style={{
          height: "150px",
          width: "150px",
        }}
        src={imageThumbnail} 
        alt={title} 
      />
      <div 
        style={{
          display: "flex",
          flexDirection: "column",
          textAlign: "left"
        }}
      >
        <span style={{ marginBottom: "10px" }}>
          <span style={{fontWeight: "bold"}}>Title:</span> {title}
        </span>
        <span style={{ marginBottom: "10px" }}>
          <span style={{fontWeight: "bold"}}>Brand:</span> {brand ? brand : "Not available"}
        </span>
        <span style={{ marginBottom: "10px" }}>
          <span style={{fontWeight: "bold"}}>Price:</span> {price ? price : "Not available"}
        </span>
        <span style={{ marginBottom: "10px" }}>
          <span style={{fontWeight: "bold"}}>Description:</span> {productDescription}
        </span>
      </div>
    </div>
  );
}

export default ProductInfo;
