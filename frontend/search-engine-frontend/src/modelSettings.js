export const modelOptions = {
  "subcategory": [
    { value: "M1", label: "M1 - LSI" },
    { value: "M2", label: "M2 - TF-IDF" },
    { value: "M2+", label: "M2+ - TF-IDF ASIN" },
  ],
  "ranking": [
    { value: "R1", label: "R1 - Baseline Ranking Algorithm" },
    { value: "R2", label: "R2 - Review Sentiment Algorithm" },
    { value: "R3", label: "R3 - Collaborative Filtering: User-Product Context", isDisabled: true },
  ]
}

export const productFilters = {
  R1: [
    { 
      label: "Top Features",
      key: "top_features",
    },
    { 
      label: "Top Value",
      key: "top_value",
    },
    { 
      label: "Top Sellers",
      key: "top_sellers",
    },
    { 
      label: "Top Ratings",
      key: "top_ratings",
    },
  ],
  R2: [
    { 
      label: "Top Matches",
      key: "top_matches",
    },
    { 
      label: "Top Quality",
      key: "top_quality",
    },
    { 
      label: "Top Value",
      key: "top_value",
    },
    { 
      label: "Top Sellers",
      key: "top_sellers",
    },
    { 
      label: "Top Ratings",
      key: "top_ratings",
    },
    { 
      label: "Top Reviews",
      key: "top_reviews",
    },
  ]
}
