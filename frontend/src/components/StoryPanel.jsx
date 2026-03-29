export default function StoryPanel({ onEnterExplore }) {
  const cards = [
    {
      title: "Real Martian terrain",
      body: "Elevation comes from the HiRISE camera in Mars orbit — Jezero Crater, where Perseverance landed. What you see is public NASA data, not an artist’s impression.",
    },
    {
      title: "Three approaches, one grid",
      body: "The terrain-optimized planner searches a slope-aware cost map (physics, energy, and hazard costs). The shortest-path chord ignores those costs. A reinforcement-learning policy rolls out on the same cells. Same world, three philosophies.",
    },
    {
      title: "Drive it",
      body: "Place start and goal, compare paths, then Drive to follow each route in 3D with orbit or follow camera — telemetry is Mars-flavored, not cosmetic.",
    },
  ];
  return (
    <div className="story-panel">
      <p className="story-kicker">NASA open data · Mars Route Zero</p>
      <p className="story-lede">
        Turn a real digital elevation model into a live question: how should a
        rover move when the shortest line on the map is a bad line on the ground?
      </p>

      <p className="story-manifesto">
        Billions of dollars and years of operations ride on route choices made
        with limited power and no repair shop. This visualization does not
        replace mission software — it makes the{" "}
        <em className="story-em">why</em> visceral: you pick two points on
        Jezero, and classical planning, a naive baseline, and a learned policy
        all answer on identical terrain.
      </p>

      <section className="story-learned" aria-label="What this demonstrates">
        <h3 className="story-subhead">What you can take away</h3>
        <ul className="story-bullets">
          <li>
            Map distance can mislead: steep or rough cells cost energy even when
            the chord looks short.
          </li>
          <li>
            The optimized planner on a cost map makes the distance–safety trade
            explicit; numbers next to each path show it.
          </li>
          <li>
            Putting classical search, RL, and a drivable 3D view in one page is
            rare — it turns abstract “mission planning” into something you can
            argue with.
          </li>
        </ul>
      </section>

      {cards.map((c) => (
        <section key={c.title} className="story-card">
          <h3>{c.title}</h3>
          <p>{c.body}</p>
        </section>
      ))}

      <p className="story-novelty">
        <strong>Novelty.</strong> Most Mars apps show a globe or a static map.
        Here the dataset, planners, and policy share one grid — so the story is
        the <em>gap</em> between strategies, not a single line on a chart.
      </p>

      <button type="button" className="story-cta" onClick={onEnterExplore}>
        Explore the terrain
      </button>
      <p className="story-footer">
        API <span className="story-mono">:8000</span> · UI{" "}
        <span className="story-mono">:5173</span> · C++ terrain + FastAPI +
        Three.js
      </p>
    </div>
  );
}
