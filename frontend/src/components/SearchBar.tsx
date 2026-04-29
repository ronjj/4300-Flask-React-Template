import { FormEvent, Ref } from "react";
import searchIcon from "../assets/mag.png";

type SearchBarProps = {
    value: string;
    onChange: (nextValue: string) => void;
    onSubmit: () => void;
    placeholder?: string;
    disabled?: boolean;
    className?: string;
    inputRef?: Ref<HTMLInputElement>;
    onFocus?: () => void;
    autoFocus?: boolean;
    showAiToggle?: boolean;
    aiMode?: boolean;
    onAiToggle?: () => void;
}

function SearchBar({
    value,
    onChange,
    onSubmit,
    placeholder = "look up the best Brazilian wingers...",
    disabled = false,
    className = "",
    inputRef,
    onFocus,
    autoFocus = false,
    showAiToggle = false,
    aiMode = false,
    onAiToggle,
}: SearchBarProps): JSX.Element {
        const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
            event.preventDefault();
            onSubmit();
        };
        return (
            <form className={`search-bar ${className}`.trim()} onSubmit={handleSubmit}>
            <img className="search-bar-icon" src={searchIcon} alt="" aria-hidden="true" />
            <input
                className="search-bar-input"
                type="text"
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                disabled={disabled}
                autoComplete="off"
                aria-label="Search players"
                ref={inputRef}
                onFocus={onFocus}
                autoFocus={autoFocus}
            />
            {showAiToggle && (
                <button
                    type="button"
                    className={`ai-toggle ${aiMode ? "ai-toggle--on" : ""}`}
                    onClick={onAiToggle}
                    aria-label={aiMode ? "AI mode on, click to disable" : "AI mode off, click to enable"}
                    aria-pressed={aiMode}
                >
                    <span className="ai-toggle-icon">✦</span>
                    <span className="ai-toggle-label">AI</span>
                </button>
            )}
            <button className="search-bar-btn" type="submit" disabled={disabled} aria-label="Submit search">
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3.5 9H14.5M14.5 9L10 4.5M14.5 9L10 13.5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
            </button>
            </form>
        );
    }
    export default SearchBar;
