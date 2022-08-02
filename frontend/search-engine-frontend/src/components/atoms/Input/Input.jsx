import React from "react";

function Input({ placeholder, className, value, style, onChange, onKeyDown }) {
  return (
    <input 
      type="text" 
      className={className}
      placeholder={placeholder}
      value={value}
      style={style}
      onChange={onChange}
      onKeyDown={onKeyDown}
    />
  );
}

export default Input;
