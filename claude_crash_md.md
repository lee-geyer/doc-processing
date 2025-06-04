# Claude Code Memory Crash Analysis & Solutions

## Crash Summary
Claude Code crashed with a JavaScript heap out of memory error during string operations. The Node.js process exceeded its memory limit and the garbage collector could not free sufficient memory to continue.

## Error Details
- **Error Type**: `FATAL ERROR: Ineffective mark-compacts near heap limit Allocation failed - JavaScript heap out of memory`
- **Location**: String operations (`v8::internal::String::SlowFlatten`)
- **Memory Usage**: ~3.9GB before crash
- **Node Version**: v22.15.0

## Root Cause Analysis
The crash occurred during:
1. String flattening operations (combining/processing large strings)
2. Property access operations that required string manipulation
3. Likely processing of large data structures or files

## Immediate Solutions

### 1. Increase Node.js Memory Limit (Recommended)
Add to your `~/.zshrc` file:
```bash
# Set Node.js memory limit to 8GB (adjust as needed)
export NODE_OPTIONS="--max-old-space-size=8192"

# For very large operations, consider 32GB:
# export NODE_OPTIONS="--max-old-space-size=32768"
```

Apply changes:
```bash
source ~/.zshrc
```

### 2. Alternative: Direct Memory Flag
Run Claude Code with increased memory:
```bash
node --max-old-space-size=8192 ~/.claude/local/claude
```

## System Context
- **Hardware**: Mac Studio with 512GB RAM
- **Available Memory**: Sufficient for much larger limits
- **Recommended Setting**: 16-32GB (`--max-old-space-size=16384` to `32768`)

## Development Recommendations

### For Current Work
1. **Identify Memory-Intensive Operations**
   - Large file processing
   - Data aggregation/transformation
   - String concatenation in loops
   - JSON parsing of large datasets

2. **Implement Streaming Approaches**
   - Process data in chunks rather than loading everything into memory
   - Use Node.js streams for file operations
   - Consider readline for large text files

3. **Memory Monitoring**
   - Use `process.memoryUsage()` to track memory consumption
   - Implement garbage collection hints with `global.gc()` (if needed)
   - Monitor with Activity Monitor during development

### Alternative Approaches
Given preference for Python development:
- **Consider Python alternatives** for memory-intensive data processing
- **Use uv environments** to isolate different memory requirements
- **Prototype in Jupyter** to test memory usage patterns before implementing in production

## Prevention Strategies

### Code Patterns to Avoid
```javascript
// Avoid: Building large strings in memory
let result = "";
for (let i = 0; i < largeArray.length; i++) {
    result += processItem(largeArray[i]);
}

// Prefer: Streaming or chunked processing
const chunks = [];
for (let i = 0; i < largeArray.length; i++) {
    chunks.push(processItem(largeArray[i]));
    if (chunks.length > 1000) {
        await processChunks(chunks);
        chunks.length = 0; // Clear array
    }
}
```

### Recommended Patterns
```javascript
// Memory-efficient file processing
const fs = require('fs');
const readline = require('readline');

const fileStream = fs.createReadStream('large-file.txt');
const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity
});

for await (const line of rl) {
    // Process line by line instead of loading entire file
    await processLine(line);
}
```

## Next Steps
1. **Apply memory limit increase** using the export command above
2. **Restart terminal/Claude Code** to apply new settings
3. **Retry the operation** that caused the crash
4. **Monitor memory usage** during execution
5. **Consider refactoring** if memory usage remains problematic

## Additional Context for Claude Code
- User prefers Python for development but working in Node.js environment
- User has extensive RAM (512GB) available
- User prefers well-documented, production-ready solutions
- User develops generative and analytical applications

Please analyze the specific operation that caused this crash and implement appropriate memory management strategies based on the context of our current work.