// src/components/Label.tsx

import React, { LabelHTMLAttributes, ReactNode } from "react";
import "./Label.css";

interface LabelProps extends LabelHTMLAttributes<HTMLLabelElement> {
  children: ReactNode;
}

const Label = React.forwardRef<HTMLLabelElement, LabelProps>(
  ({ children, className = "", ...props }, ref) => {
    return (
      <label ref={ref} className={`label ${className}`} {...props}>
        {children}
      </label>
    );
  }
);

Label.displayName = "Label";

export default Label;
