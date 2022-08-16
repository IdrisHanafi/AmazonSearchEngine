import React from "react";

import "./Filters.css"
import { productFilters } from "../../../modelSettings";

// import componenets
function Filters({ selectedFilterOptions, selectedFilter, onClick }) {
  let productFilter = productFilters[selectedFilterOptions];

  return (
    <div 
      style={{
        display: "flex",
        justifyContent: "center",
        width: "fit-content",
        minWidth: "60%",
        maxWidth: "70%",
      }}
    >
      {productFilter.map((item, index) => {
        return (
          <div 
            style={{
              ...(index === 0 ? {
                borderTopLeftRadius: "10px",
                borderBottomLeftRadius: "10px",
              } : {}),
              ...(index === productFilter.length - 1 ? {
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
  selectedFilterOptions: "R1"
}

export default Filters;
