import React from "react";

import "./CheckBox.css"

function CheckBox({ items, selectedIndex, onClick }) {
  return (
    <div 
      style={{
        width: "fit-content",
        minWidth: "60%",
        maxWidth: "70%",
      }}
    >
      {items.map((item, index) => {
        return (
          <div 
            key={item.index}
            onClick={() => onClick(index)} 
          className={
            index === selectedIndex ? "customCheckBoxSelected" : "customCheckBox"
            }
          > 
            {item.label}
          </div>

        )
      })}
    </div>
  );
}

CheckBox.defaultProps = {
  items: [
    { 
      label: "Subcategory 1",
      index: 100,
    },
    { 
      label: "Subcategory 2",
      index: 101,
    },
    { 
      label: "Subcategory 3",
      index: 102,
    },
  ]
}

export default CheckBox;
