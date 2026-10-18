const q = '<|"|>';

function quoted(value: string) {
  return `${q}${value}${q}`;
}

function edge(
  id: string,
  from: string,
  fromPort: string,
  to: string,
  toPort: string,
  label: string,
) {
  return `{id:${quoted(id)},from:{nodeId:${quoted(from)},port:${quoted(fromPort)}},to:{nodeId:${quoted(to)},port:${quoted(toPort)}},label:${quoted(label)}}`;
}

function node(id: string, type: string, text: string) {
  return `{id:${quoted(id)},type:${quoted(type)},text:${quoted(text)}}`;
}

/** Gemma follow-up that quotes strings with `<|"|>` and never reaches the native tool parser. */
export function promisesFollowUpCall() {
  const edges = [
    edge('e1', 'start', 'top', 'concept1', 'left', 'Define Concept'),
    edge('e2', 'concept1', 'bottom', 'concept2', 'left', 'Leads to States'),
    edge('e3', 'concept2', 'right', 'concept3', 'top', 'Requires Creation'),
    edge('e4', 'concept3', 'bottom', 'concept4', 'left', 'Creation leads to Usage'),
    edge('e5', 'concept4', 'right', 'concept5', 'top', 'Usage enables Chaining'),
    edge('e6', 'concept5', 'bottom', 'concept6', 'left', 'Chaining involves Error Handling'),
    edge('e7', 'concept6', 'bottom', 'end', 'top', 'Final Outcome'),
  ];
  const nodes = [
    node('start', 'rounded-rect', 'Start: Understanding JavaScript Promises'),
    node('concept1', 'diamond', 'What is a Promise?'),
    node('concept2', 'rect', 'States of a Promise (Pending, Fulfilled, Rejected)'),
    node('concept3', 'rect', 'Creating a Promise (new Promise())'),
    node('concept4', 'rect', 'Consuming a Promise (then(), catch(), finally)'),
    node('concept5', 'rect', 'Promise Chaining (.then())'),
    node('concept6', 'rect', 'Error Handling (catch())'),
    node('end', 'rounded-rect', 'End: Asynchronous Flow Complete'),
  ];
  return `call:set_flowchart_document{document:{edges:[${edges.join(',')}],nodes:[${nodes.join(',')}]}}`;
}

export function promisesFollowUpDump() {
  const call = promisesFollowUpCall();
  return `Failed to parse tool calls from code block: ${call} full response: <|tool_call>${call}<tool_call|> error: Failed to parse FC tool calls: FallThrough(ParseCancelledError(InputMismatchError(InputMisMatchError { base: BaseRecognitionError { message: "", offending_token: GenericToken { token_type: -1, channel: 0, start: 1921, stop: 1920, token_index: 365, line: 1, column: 1921, text: "", read_only: false }, offending_state: 32, states_stack: [18, 12] } })))`;
}
