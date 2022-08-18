import ReactLoading from 'react-loading';
 
function Loading({ isLoading }) {
  return isLoading && (
    <div 
      style={{
        display: "flex",
        justifyContent: "center"
      }}
    >
      <ReactLoading type="spin" color="#000"/>
    </div>
  );
}
export default Loading;
