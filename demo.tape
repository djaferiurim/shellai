# ShellAI demo recording — render with: vhs demo.tape
# Install VHS: https://github.com/charmbracelet/vhs
# Produces demo.gif for the README.

Output demo.gif

Set Shell "powershell"
Set FontSize 18
Set Width 1100
Set Height 640
Set Theme "Catppuccin Mocha"
Set TypingSpeed 60ms
Set Padding 20

# --- intro ---
Type "# 🐚 ShellAI — chat with any LLM from your terminal"
Sleep 1s
Enter
Sleep 500ms

# --- one-shot question ---
Type 'ai "explain monads like I am five"'
Sleep 800ms
Enter
Sleep 6s

# --- a persona ---
Type 'ai --persona pirate "what is git?"'
Sleep 800ms
Enter
Sleep 5s

# --- the coding agent ---
Type 'ai agent --yolo "create hello.py that prints a greeting and run it"'
Sleep 800ms
Enter
Sleep 8s

# --- chat with your files (local RAG) ---
Type 'ai index README.md'
Sleep 500ms
Enter
Sleep 4s
Type 'ai ask "which LLM providers does ShellAI support?"'
Sleep 800ms
Enter
Sleep 9s

Sleep 2s
