function ProductInfo({ title, imageURL }) {
  let imageThumbnail = null;
  if (imageURL.length > 0) {
    imageThumbnail = imageURL[0];
  }

  return (
    <div>
      <img 
        style={{
          height: "100px",
          width: "100px",
        }}
        src={imageThumbnail} 
        alt={title} 
      />
    </div>
  );
}

export default ProductInfo;
