import React from "react";

function Input({ placeholder, className, value, style, onChange, onKeyPress }) {
  return (
    <input type="text" className={className} placeholder={placeholder} value={value} style={style} onChange={onChange} onKeyPress={onKeyPress}/>
  );
}

export default Input;
