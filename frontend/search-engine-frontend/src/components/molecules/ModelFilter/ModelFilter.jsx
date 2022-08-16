import Dropdown from "../../atoms/Dropdown/Dropdown";

import { modelOptions } from "../../../modelSettings";

function ModelFilter({
  label,
  options,
  selectedValue,
  onChange 
}) {

  const handleTypeSelect = e => {
    onChange(e.value);
  }

  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      padding: "10px",
    }}>
      <p style={{ paddingRight: "10px" }}>
        {label}
      </p>
      <Dropdown
        options={modelOptions[options]} 
        value={modelOptions[options].filter(function(option) {
          return option.value === selectedValue;
        })} 
        onChange={handleTypeSelect} 
      />
    </div>
  );
}

ModelFilter.defaultProps = {
  options: "subcategory"
};

export default ModelFilter;
