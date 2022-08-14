import Dropdown from "../../atoms/Dropdown/Dropdown";

const modelOptions = {
  "subcategory": [
    { value: "M1", label: "M1 - LSI" },
    { value: "M2", label: "M2 - TF-IDF" },
    { value: "M2+", label: "M2+ - TF-IDF ASIN" },
  ],
  "ranking": [
    { value: "R1", label: "R1 - Baseline Ranking Algorithm" },
    { value: "R2", label: "R2 - Review Sentiment Algorithm" },
    { value: "R3", label: "R3 - User-Product Context" },
  ]
}

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
