// src/components/Button.tsx

import React, { ButtonHTMLAttributes, ReactNode } from "react";
import "./Button.css";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline";
  fullWidth?: boolean;
  children: ReactNode;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "default", fullWidth = false, children, className = "", ...props }, ref) => {
    const variantClass = variant === "outline" ? "button-outline" : "button-default";
    const widthClass = fullWidth ? "button-fullWidth" : "";
    return (
      <button
        ref={ref}
        className={`button ${variantClass} ${widthClass} ${className}`}
        {...props}
      >
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";

export default Button;
