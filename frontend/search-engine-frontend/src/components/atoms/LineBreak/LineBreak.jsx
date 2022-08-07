function LineBreak({ width }) {
  return (
    <hr
      style={{
        background: 'black',
        height: '1px',
        width: width,
        marginTop: '20px',
      }}
    />
  );
}

LineBreak.defaultProps = {
  width: '70%'
}

export default LineBreak;
