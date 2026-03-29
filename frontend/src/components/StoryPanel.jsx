/** PRD §9 — short narrative before explore mode */
export default function StoryPanel({ onEnterExplore }) {
  const cards = [
    {
      title: "This is Jezero Crater",
      body: "The terrain you see comes from a real NASA elevation grid (HiRISE-class DTM in production; synthetic here for dev). Every hill and hollow is data, not decoration.",
    },
    {
      title: "Why not a straight line?",
      body: "Distance is cheap; climbing loose slopes is expensive. Planners minimize a terrain cost—similar in spirit to how Mars rovers trade safety and energy.",
    },
    {
      title: "Try three paths",
      body: "A* follows the cost map, straight line ignores hazards, and the RL policy learns from the same grid. Compare distance versus energy-style scores in the panel.",
    },
  ];
  return (
    <div className="story-panel">
      <p className="story-kicker">Mars Route Zero</p>
      {cards.map((c) => (
        <section key={c.title} className="story-card">
          <h3>{c.title}</h3>
          <p>{c.body}</p>
        </section>
      ))}
      <button type="button" className="story-cta" onClick={onEnterExplore}>
        Explore terrain
      </button>
    </div>
  );
}
