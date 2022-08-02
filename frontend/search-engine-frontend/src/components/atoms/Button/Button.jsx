import React from "react";

// import styles
import "./Button.css";

function Button({children, text, className, style, onClick}) {
  return (
    <button className={className} style={style} onClick={onClick}>
      {children || text}
    </button>
  )
}

Button.defaultProps = {
  className: "default",
};

export default Button;
