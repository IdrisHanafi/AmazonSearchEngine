import Select from 'react-select'

const options = [
  { value: 'chocolate', label: 'Chocolate' },
  { value: 'strawberry', label: 'Strawberry' },
  { value: 'vanilla', label: 'Vanilla' }
]

function Dropdown({
  options,
  value,
  onChange 
}) {
  return (
    <Select 
      options={options} 
      value={value} 
      onChange={onChange} 
    />
  );
}

Dropdown.defaultProps = {
  options: options
}

export default Dropdown;
