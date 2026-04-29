// src/components/Input.tsx

import React, { InputHTMLAttributes } from "react";
import "./Input.css";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className = "", ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={`input ${className}`}
        {...props}
      />
    );
  }
);

Input.displayName = "Input";

export default Input;
