import React from "react";

import "./Filters.css"
 
// import componenets
function Filters({ items, selectedFilter, onClick }) {
  return (
    <div 
      style={{
        display: "flex",
        justifyContent: "space-between",
        width: "fit-content",
        minWidth: "60%",
        maxWidth: "70%",
      }}
    >
      {items.map((item, index) => {
        return (
          <div 
            style={{
              ...(index === 0 ? {
                borderTopLeftRadius: "10px",
                borderBottomLeftRadius: "10px",
              } : {}),
              ...(index === items.length - 1 ? {
                borderTopRightRadius: "10px",
                borderBottomRightRadius: "10px",
              } : {}),
            }}
            key={item.key}
            onClick={() => onClick(item.key)} 
            className={
              item.key === selectedFilter ?
                "filterButtonSelected" :
                "filterButtons"
            }
          > 
            {item.label}
          </div>

        )
      })}
    </div>
  );
}

Filters.defaultProps = {
  items: [
    { 
      label: "Top Features",
      key: "top_features",
    },
    { 
      label: "Top Value",
      key: "top_value",
    },
    { 
      label: "Top Sellers",
      key: "top_sellers",
    },
    { 
      label: "Top Ratings",
      key: "top_ratings",
    },
  ]
}

export default Filters;
