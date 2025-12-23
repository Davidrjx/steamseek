import nltk
nltk.download('punkt_tab', quiet=True)

import json
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import asyncio
from openai import OpenAI
import signal
import sys
from dotenv import load_dotenv
from review_filtering import ReviewFilter

@dataclass
class ProcessingState:
    """Track processing state for resumability"""
    last_processed_game: Optional[str] = None
    processed_count: int = 0
    failed_games: List[str] = None
    start_time: float = None

    def __post_init__(self):
        self.failed_games = self.failed_games or []
        self.start_time = self.start_time or time.time()

class GameSummarizer:
    def __init__(self, 
                 input_file: str,
                 output_file: str,
                 state_file: str,
                 openrouter_key: Optional[str] = None,
                 model: str = None):
        # Save file paths and settings to instance variables
        self.input_file = input_file
        self.output_file = output_file
        self.state_file = state_file
        self.model = model if model else "mistralai/mistral-nemo"

        # Load environment variables
        load_dotenv()
        
        # Get API key with fallback to environment variable
        # api_key = openrouter_key or os.getenv('OPENROUTER_API_KEY')
        api_key = os.getenv('AIGC_API_KEY') if os.getenv('AIGC_API_KEY') else os.getenv('OPENROUTER_API_KEY')
        api_base_url = os.getenv('AIGC_OPENAI_BASE_URL') if os.getenv('AIGC_OPENAI_BASE_URL') else os.getenv('OPENROUTER_BASE_URL')
        if not api_key:
            raise ValueError("Fuxi AIGC or OpenAI API key and conresponding base url must be provided")
            
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url = api_base_url,
            api_key=api_key
        )
        
        # Initialize review filter
        self.review_filter = ReviewFilter()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('game_summarizer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load or initialize state
        self.state = self._load_state()
        
        # Setup interrupt handling
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _load_state(self) -> ProcessingState:
        """Load processing state from file or create new state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return ProcessingState(**data)
            except Exception as e:
                self.logger.warning(f"Error loading state file: {e}. Starting fresh.")
        return ProcessingState()

    def _save_state(self):
        """Save current processing state to file."""
        with open(self.state_file, 'w') as f:
            json.dump({
                'last_processed_game': self.state.last_processed_game,
                'processed_count': self.state.processed_count,
                'failed_games': self.state.failed_games,
                'start_time': self.state.start_time
            }, f)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals by saving state and exiting gracefully."""
        self.logger.info("\nInterrupt received. Saving state and exiting...")
        self._save_state()
        stats = self._get_statistics()
        self.logger.info("Final Statistics:")
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")
        sys.exit(0)

    def _filter_reviews(self, reviews: List[Dict]) -> List[Dict]:
        """
        Filter reviews using the ReviewFilter class.
        Returns the top filtered reviews sorted by quality score.
        """
        return self.review_filter.filter_reviews(reviews)

    def _prepare_summary_prompt(self, game_data: Dict) -> str:
        """
        Prepare prompt for the LLM to generate game summary with explicit thinking process.
        Includes game metadata and filtered reviews.
        """
        # Get basic game info
        name = game_data.get('name', 'Unknown Game')
        description = game_data.get('short_description', '')
        detailed_desc = game_data.get('detailed_description', '')
        genres = ', '.join(game_data.get('genres', [])) if game_data.get('genres') else 'Unknown'

        # Handle release_date which can be string or dict
        release_date_data = game_data.get('release_date', 'Unknown')
        if isinstance(release_date_data, dict):
            release_date = release_date_data.get('date', 'Unknown')
        else:
            release_date = release_date_data if release_date_data else 'Unknown'

        developers = ', '.join(game_data.get('developers', [])) if game_data.get('developers') else 'Unknown'

        # Get review statistics
        reviews_list = game_data.get('reviews', [])
        total_reviews = len(reviews_list)
        positive_reviews = sum(1 for r in reviews_list if r.get('voted_up', False))
        negative_reviews = total_reviews - positive_reviews
        positive_ratio = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 0

        # Filter and get top reviews
        filtered_reviews = self._filter_reviews(reviews_list)
        top_reviews = sorted(filtered_reviews,
                           key=lambda x: x.get('quality_score', 0),
                           reverse=True)[:8]  # Use top 8 highest quality reviews

        # Construct enhanced professional analysis prompt
        prompt = f"""You are a professional video game industry analyst conducting a comprehensive game evaluation. Your task is to produce a detailed analytical report following professional industry standards.

## ANALYSIS FRAMEWORK

<thinking>

### SECTION 1: TECHNICAL & MECHANICAL ANALYSIS
**1.1 Core Gameplay Architecture:**
- Describe the fundamental gameplay loop and its cycle structure
- Identify primary mechanical pillars (e.g., combat system, resource management, progression mechanics)
- Analyze mechanical depth: surface-level vs. mastery-level gameplay
- Evaluate mechanical innovation: what's unique vs. what's derivative from genre conventions

**1.2 Systems Integration:**
- How do different game systems interconnect (e.g., economy → progression → combat)?
- Identify feedback loops (positive reinforcement, balancing mechanisms)
- Assess system complexity: appropriate for target audience or over/under-designed?
- Note any mechanical friction points or synergy opportunities

**1.3 Technical Implementation:**
- Based on reviews, assess performance, optimization, and technical stability
- Identify any significant technical innovations (engine, graphics, physics, AI)
- Note technical limitations or issues affecting gameplay

---

### SECTION 2: PLAYER SENTIMENT & COMMUNITY ANALYSIS
**2.1 Quantitative Assessment:**
- Total reviews analyzed: {total_reviews}
- Positive sentiment: {positive_reviews} ({positive_ratio:.1f}%)
- Negative sentiment: {negative_reviews} ({100-positive_ratio:.1f}%)

**2.2 Qualitative Thematic Analysis:**
- POSITIVE THEMES: Identify recurring praise patterns
  * What specific features/mechanics are most appreciated?
  * What emotional responses are mentioned (excitement, satisfaction, immersion)?
  * Which design choices resonate with the community?

- NEGATIVE THEMES: Identify recurring criticism patterns
  * What are the primary pain points?
  * Are criticisms about design choices or implementation quality?
  * Which features disappoint or frustrate players?

**2.3 Sentiment Segmentation:**
- Casual vs. hardcore player perspectives
- New player vs. veteran player experiences
- Are certain player types better served than others?

**2.4 Community Consensus:**
- What is the "majority opinion" on this game?
- Are there polarizing elements causing divided opinions?
- Is sentiment trajectory improving or declining over time?

---

### SECTION 3: MARKET POSITIONING & COMPETITIVE ANALYSIS
**3.1 Genre Context:**
- Genre classification: {genres}
- Where does this game sit within its genre spectrum?
- What sub-genre or hybrid elements are present?

**3.2 Unique Selling Propositions (USPs):**
- What makes this game stand out in a crowded market?
- Which features would attract players away from competitors?
- Are the USPs meaningful innovations or superficial differentiators?

**3.3 Target Audience Identification:**
- Who is this game designed for (demographic, psychographic, player archetype)?
- Does the execution match the intended audience?
- Are there underserved or unintended audiences finding value?

**3.4 Competitive Positioning:**
- What are comparable titles in the market?
- Does this game compete on innovation, execution, price, or accessibility?
- What market gaps does it fill or fail to address?

---

### SECTION 4: EXPERIENCE DESIGN & PLAYER JOURNEY
**4.1 Onboarding & Accessibility:**
- How welcoming is the game to new players?
- Learning curve assessment: gentle, moderate, steep?
- Tutorial and guidance systems effectiveness

**4.2 Pacing & Content Structure:**
- Early game experience (first 2-5 hours)
- Mid-game progression and engagement maintenance
- Late-game/endgame content depth
- Replayability factors

**4.3 Emotional Arc & Engagement:**
- What emotional journey does the game create?
- Peak moments and low points in player experience
- Engagement retention strategies (hooks, rewards, progression)

**4.4 Social & Multiplayer Dimensions:**
- Single-player, multiplayer, or hybrid focus?
- Social interaction design (competitive, cooperative, emergent)
- Community features and their effectiveness

---

### SECTION 5: QUALITY & POLISH ASSESSMENT
**5.1 Production Values:**
- Art direction and visual presentation quality
- Audio design (music, sound effects, voice acting)
- UI/UX design effectiveness
- Narrative quality (if applicable)

**5.2 Content Volume vs. Quality:**
- How much content is available?
- Is content depth prioritized over breadth, or vice versa?
- Value proposition: content hours vs. price point

**5.3 Post-Launch Support:**
- Evidence of ongoing updates, patches, or expansions
- Developer responsiveness to community feedback
- Long-term content roadmap signals

---

### SECTION 6: SYNTHESIS & STRATEGIC INSIGHTS
**6.1 Core Strengths (Priority-Ordered):**
- List 3-5 primary strengths with specific evidence
- Which strengths are most defensible/unique?

**6.2 Core Weaknesses (Priority-Ordered):**
- List 3-5 primary weaknesses with specific evidence
- Which weaknesses are most addressable vs. fundamental?

**6.3 Design Philosophy Assessment:**
- What is the apparent design vision?
- How successfully is that vision executed?
- Are there contradictions between design goals and implementation?

**6.4 Recommendations:**
- For potential players: Who should buy this? Who should avoid it?
- For developers (hypothetical): What are the top 3 improvement priorities?

</thinking>

---

<summary>
## 游戏综合分析报告

Based on the comprehensive analysis above, create a detailed **Chinese-language analytical summary (600-800 words)** structured as follows:

### 【游戏概述】
- 游戏基本信息和核心定位（2-3句话）
- 类型归属和市场定位

### 【核心玩法机制】
- 详细描述主要游戏循环和机制系统（150-200字）
- 分析机制深度和创新性
- 评估系统整合质量

### 【创新特色与差异化优势】
- 具体列举该游戏的独特卖点（100-150字）
- 与同类游戏的对比优势
- 技术或设计层面的突破

### 【玩家体验与社区反馈】
- 基于{total_reviews}条评测的定量分析（好评率{positive_ratio:.1f}%）
- 玩家高频赞扬的具体要素
- 玩家普遍批评的具体问题
- 体验曲线分析（上手难度、中期深度、后期内容）

### 【市场表现与受众适配】
- 目标受众画像
- 市场竞争力评估
- 适合/不适合的玩家类型

### 【综合评价】
- 制作质量与完成度评估
- 核心优势总结（3-4点，具体化）
- 主要不足总结（3-4点，具体化）
- 整体推荐指数和理由

**写作要求：**
1. 使用专业、客观的分析语言，避免营销话术
2. 所有评价必须基于具体证据（来自描述或玩家评测）
3. 量化信息优先（如"8成玩家认为..."、"核心玩法循环约X小时"）
4. 正负面评价平衡，如实反映游戏优缺点
5. 结论明确，给出清晰的适用人群建议
</summary>

---

## SOURCE DATA

**Game Title:** {name}
**Genre:** {genres}
**Developer:** {developers}
**Release Date:** {release_date}

**Official Short Description:**
{description}

**Detailed Description:**
{detailed_desc}

**Community Reviews Analysis ({total_reviews} total reviews, {positive_ratio:.1f}% positive):**
"""

        for i, review in enumerate(top_reviews, 1):
            review_text = review.get('review', '').strip()
            is_positive = review.get('voted_up', False)
            playtime = review.get('author', {}).get('playtime_forever', 0)
            playtime_hours = playtime // 60 if playtime else 0
            sentiment = "👍 POSITIVE" if is_positive else "👎 NEGATIVE"
            quality_score = review.get('quality_score', 0)

            prompt += f"\n--- Review {i} ({sentiment}, Playtime: {playtime_hours}h, Quality Score: {quality_score:.2f}) ---\n{review_text}\n"

        prompt += "\n---\n\n**TASK:** Complete the <thinking> analysis following all 6 sections above, then write the <summary> Chinese report following the specified structure. Be thorough, specific, and evidence-based."

        return prompt

    async def _generate_summary(self, game_data: Dict) -> Optional[Dict[str, str]]:
        """
        Generate summary for a single game using LLM with thinking process.
        Returns a dict with 'summary' and 'thinking_process', or None if generation fails.
        """
        try:
            prompt = self._prepare_summary_prompt(game_data)
            messages = [
                {"role": "system", "content": """You are a senior video game industry analyst with expertise in:
- Game design analysis and mechanical evaluation
- Player psychology and engagement metrics
- Market trends and competitive positioning
- Technical assessment (graphics, performance, UX)
- Community sentiment analysis

Your reports are known for being:
✓ Thoroughly researched and evidence-based
✓ Balanced and objective (not promotional)
✓ Specific and actionable (avoiding vague generalizations)
✓ Professional yet accessible in tone

Always structure your analysis with clear <thinking> and <summary> sections. Your thinking process should be comprehensive, following all analytical frameworks provided. Your summaries should be detailed, specific, and professionally written."""},
                {"role": "user", "content": prompt}
            ]

            self.logger.info(f"Requesting detailed analysis report for: {game_data.get('name')}")

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,  # Slightly lower for more focused analysis
                max_tokens=8000  # Significantly increased for comprehensive report
            )

            content = completion.choices[0].message.content

            # Extract thinking process and summary
            thinking_process = None
            summary = None

            if "<thinking>" in content and "</thinking>" in content:
                thinking_start = content.find("<thinking>") + 10
                thinking_end = content.find("</thinking>")
                thinking_process = content[thinking_start:thinking_end].strip()
                self.logger.info(f"✓ Extracted thinking process ({len(thinking_process)} characters)")

            if "<summary>" in content and "</summary>" in content:
                summary_start = content.find("<summary>") + 9
                summary_end = content.find("</summary>")
                summary = content[summary_start:summary_end].strip()
                self.logger.info(f"✓ Extracted summary ({len(summary)} characters)")
            else:
                # Fallback: if no tags, use all content after thinking section
                if thinking_process and "</thinking>" in content:
                    summary = content[content.find("</thinking>") + 12:].strip()
                else:
                    # If no structure at all, use entire content as summary
                    summary = content

            return {
                "summary": summary,
                "thinking_process": thinking_process
            }

        except Exception as e:
            self.logger.error(f"Error generating summary for {game_data.get('name')}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _get_statistics(self) -> Dict[str, Any]:
        """Calculate processing statistics."""
        elapsed = time.time() - self.state.start_time
        return {
            'Processed Games': self.state.processed_count,
            'Failed Games': len(self.state.failed_games),
            'Elapsed Time': f"{elapsed/3600:.2f} hours",
            'Processing Rate': f"{self.state.processed_count/elapsed:.2f} games/second" if elapsed > 0 else "N/A"
        }

    async def process_games(self):
        """Main processing loop processing one game at a time with resumability."""
        try:
            with open(self.input_file, 'r') as infile:
                for line_num, line in enumerate(infile, 1):
                    try:
                        game_data = json.loads(line)
                        game_id = game_data.get('appid')
                        
                        # Skip if already processed
                        if (self.state.last_processed_game and 
                            line_num <= self.state.processed_count):
                            continue
                        
                        result = await self._generate_summary(game_data)

                        if result:
                            # Build a simplified output containing key metadata
                            output_data = {
                                "appid": game_data.get("appid"),
                                "name": game_data.get("name"),
                                "short_description": game_data.get("short_description"),
                                "ai_summary": result.get("summary"),
                                "thinking_process": result.get("thinking_process"),
                                "summary_generated_at": datetime.now().isoformat(),
                                "summary_model": self.model
                            }
                            
                            # Write to output file
                            with open(self.output_file, 'a') as outfile:
                                json.dump(output_data, outfile)
                                outfile.write('\n')
                            
                            self.logger.info(f"Successfully processed {game_data.get('name')} ({game_id})")
                            
                            # Update state
                            self.state.last_processed_game = game_id
                            self.state.processed_count += 1
                        else:
                            self.logger.error(f"Failed to generate summary for {game_data.get('name')} ({game_id})")
                            self.state.failed_games.append(game_id)
                        
                        # Save state after each game
                        self._save_state()
                        
                        # Optionally, wait a short delay to respect rate limits
                        await asyncio.sleep(1)
                    
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing game: {e}")
                        continue
            
            self.logger.info("Processing completed!")

        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            self._save_state()
            raise

    async def process_single_game(self, target_appid: int):
        """Process a single game by appid."""
        self.logger.info(f"Looking for game with appid {target_appid}...")

        try:
            with open(self.input_file, 'r') as infile:
                for line in infile:
                    try:
                        game_data = json.loads(line)
                        game_id = game_data.get('appid')

                        # Convert to int for comparison
                        if isinstance(game_id, str):
                            game_id = int(game_id)

                        if game_id != target_appid:
                            continue

                        # Found the game
                        self.logger.info(f"Found game: {game_data.get('name')} (appid: {game_id})")

                        result = await self._generate_summary(game_data)

                        if result:
                            output_data = {
                                "appid": game_id,
                                "name": game_data.get("name"),
                                "short_description": game_data.get("short_description"),
                                "ai_summary": result.get("summary"),
                                "thinking_process": result.get("thinking_process"),
                                "summary_generated_at": datetime.now().isoformat(),
                                "summary_model": self.model
                            }

                            # Append to output file
                            with open(self.output_file, 'a') as outfile:
                                json.dump(output_data, outfile)
                                outfile.write('\n')

                            self.logger.info(f"Successfully generated summary for {game_data.get('name')} ({game_id})")
                            print(f"\n{'='*60}")
                            print(f"AI Summary for {game_data.get('name')}:")
                            print(f"{'='*60}")

                            # Display thinking process if available
                            if result.get("thinking_process"):
                                print("\n[THINKING PROCESS]")
                                print("-" * 60)
                                print(result.get("thinking_process")[:500])  # First 500 chars
                                if len(result.get("thinking_process")) > 500:
                                    print(f"\n... (+ {len(result.get('thinking_process')) - 500} more characters)")
                                print("-" * 60)

                            print("\n[FINAL SUMMARY]")
                            print(result.get("summary"))
                            print(f"{'='*60}")
                            print(f"Summary saved to {self.output_file}")
                            return True
                        else:
                            self.logger.error(f"Failed to generate summary for {game_data.get('name')}")
                            return False

                    except json.JSONDecodeError:
                        continue

            self.logger.error(f"Game with appid {target_appid} not found in {self.input_file}")
            return False

        except Exception as e:
            self.logger.error(f"Error processing single game: {e}")
            raise

async def main():
    """Script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate game summaries using OpenRouter LLM")
    parser.add_argument("--input", default="data/steam_games_data.jsonl",
                        help="Input JSONL file path")
    parser.add_argument("--output", default="data/game_summaries.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--state", default="data/summarizer_state.json",
                        help="State file path for resuming")
    parser.add_argument("--api-key", required=False,
                        help="OpenRouter API key")
    parser.add_argument("--model",
                        help="Model identifier to use")
    parser.add_argument("--appid",
                        type=int,
                        help="Only process a single game by appid (e.g., --appid 219740)")

    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    summarizer = GameSummarizer(
        input_file=args.input,
        output_file=args.output,
        state_file=args.state,
        openrouter_key=args.api_key,
        model=args.model
    )

    # Process single game or all games
    if args.appid:
        await summarizer.process_single_game(args.appid)
    else:
        await summarizer.process_games()

if __name__ == "__main__":
    asyncio.run(main())
