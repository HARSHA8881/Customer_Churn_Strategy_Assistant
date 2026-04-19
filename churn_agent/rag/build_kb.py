import os
import chromadb
from sentence_transformers import SentenceTransformer

def get_documents():
    return [
        # Loyalty program design for banking customers
        "Loyalty program design for banking customers should focus on rewarding tenure. Customers over 5 years receive a 0.5% bonus on savings [Source: Retention Handbook, Section 1A]",
        "Banking loyalty programs must integrate daily transactions. Offering 1 point per $10 spent on debit cards increases retention by 14% [Source: Retention Handbook, Section 1B]",
        "Tiered loyalty programs ensure high-value banking clients feel exclusive. Platinum tiers should include free international wire transfers [Source: Retention Handbook, Section 1C]",
        "Gamification in loyalty programs, such as monthly transaction milestones, reduces passive churn among younger demographics [Source: Retention Handbook, Section 1D]",
        "A loyalty program for banking must explicitly communicate benefits. Unused rewards are a leading indicator of potential churn [Source: Retention Handbook, Section 1E]",

        # Personalized offer generation based on customer tenure
        "Personalized offer generation must account for customer tenure. Customers with 1-2 years tenure prefer fee waivers over rate hikes [Source: Retention Handbook, Section 2A]",
        "For clients with 5+ years of tenure, personalized mortgage refinance offers reduce churn probability by 22% [Source: Retention Handbook, Section 2B]",
        "Tenure-based personalization should trigger automatically at anniversaries. A simple 'thank you' email with a modest perk retains passively [Source: Retention Handbook, Section 2C]",
        "Customers with high tenure but declining balances should receive personalized high-yield checking account offers [Source: Retention Handbook, Section 2D]",
        "Avoid generic promotions for high-tenure clients. If a 10-year client receives a 'new customer' promo, trust drops significantly [Source: Retention Handbook, Section 2E]",

        # Proactive outreach for high-value at-risk customers
        "Proactive outreach for high-value at-risk customers requires a phone call from a dedicated relationship manager, not an automated email [Source: Retention Handbook, Section 3A]",
        "High-value clients showing sudden balance drops must be contacted within 48 hours for a financial wellness check-in [Source: Retention Handbook, Section 3B]",
        "When conducting proactive outreach to wealthy individuals, frame the conversation around wealth preservation and exclusive market insights [Source: Retention Handbook, Section 3C]",
        "At-risk indicators for high-value clients include abrupt cancellation of automatic transfers. Outreach should offer concierge financial planning [Source: Retention Handbook, Section 3D]",
        "A successful proactive outreach script for VIPs starts by acknowledging their importance and asking for direct feedback on services [Source: Retention Handbook, Section 3E]",

        # Digital engagement tactics for passive customers
        "Digital engagement tactics for passive customers should start with app push notifications highlighting unused features like budgeting tools [Source: Retention Handbook, Section 4A]",
        "Passive customers often churn silently. Sending a monthly digital summary of their spending habits increases login frequency by 30% [Source: Retention Handbook, Section 4B]",
        "To engage passive digital users, offer a one-time login bonus like a $5 coffee card to establish a habit of checking their banking app [Source: Retention Handbook, Section 4C]",
        "Interactive digital widgets, such as 'round-up savings' toggles, dramatically increase daily engagement for otherwise passive accounts [Source: Retention Handbook, Section 4D]",
        "Passive clients respond well to personalized digital quiz campaigns. E.g., 'What kind of saver are you?' increases app session times [Source: Retention Handbook, Section 4E]",

        # Fee waiver and product upgrade strategies
        "Fee waiver strategies must be deployed selectively. Waiving a late fee for a first-time offender prevents reactionary emotional churn [Source: Retention Handbook, Section 5A]",
        "When an at-risk customer complains about account maintenance fees, agents have a $50 discretionary authority to waive them instantly [Source: Retention Handbook, Section 5B]",
        "Upgrading a customer to a premium account for free for 6 months often prevents churn while cross-selling higher tier benefits [Source: Retention Handbook, Section 5C]",
        "Instead of simple fee waivers, frame the interaction as a 'courtesy product upgrade' that intrinsically lacks the disputed fees [Source: Retention Handbook, Section 5D]",
        "Fee waivers should always be accompanied by education on how the customer can avoid the fee in the future, establishing a partnership [Source: Retention Handbook, Section 5E]",

        # Customer service intervention scripts
        "Customer service intervention scripts must prioritize active listening. Begin with: 'I understand why you are frustrated with this charge.' [Source: Retention Handbook, Section 6A]",
        "When a customer explicitly threatens to close their account, the script should pivot immediately to the Retention Escalation Team [Source: Retention Handbook, Section 6B]",
        "For intervention scripts concerning rate complaints: 'While I cannot change the baseline rate, I can offer you a promotional 6-month buffer.' [Source: Retention Handbook, Section 6C]",
        "Effective intervention scripts avoid corporate jargon. Say 'Let's fix this together' instead of 'Our policy dictates that we can resolve this.' [Source: Retention Handbook, Section 6D]",
        "If a customer complains about branch closures, the intervention script should highlight mobile check deposit and refund ATM fees for a year [Source: Retention Handbook, Section 6E]",

        # Churn prevention for customers with complaints
        "Churn prevention for customers with complaints relies on swift resolution. Complaints resolved within 24 hours yield higher loyalty than non-complainers [Source: Retention Handbook, Section 7A]",
        "Customers with multiple unresolved complaints are in the Critical Risk tier. A manager must personally oversee their ticket resolution [Source: Retention Handbook, Section 7B]",
        "A follow-up call 7 days after a resolved complaint is mandatory for churn prevention. It shows the bank cares about the long-term relationship [Source: Retention Handbook, Section 7C]",
        "Log all customer complaints into a central CRM. Repeated complaints about the same UI bug indicate systemic churn risks needing IT intervention [Source: Retention Handbook, Section 7D]",
        "Offering a tangible apology, such as a temporary APY boost on savings, is more effective at preventing churn than a strictly verbal apology [Source: Retention Handbook, Section 7E]",

        # Win-back campaigns for recently churned users
        "Win-back campaigns for recently churned users should occur at the 30-day, 90-day, and 180-day marks. The 90-day mark is surprisingly effective [Source: Retention Handbook, Section 8A]",
        "To win back a churned user, offer a cash incentive ($200+) for direct deposit setup, specifically acknowledging their past relationship [Source: Retention Handbook, Section 8B]",
        "Churned users often leave for lower fees. Win-back messaging should highlight new, fee-free products introduced since their departure [Source: Retention Handbook, Section 8C]",
        "Exit surveys are crucial for win-back campaigns. Tailor the win-back offer directly to the stated reason for leaving in the exit survey [Source: Retention Handbook, Section 8D]",
        "Win-back campaigns must lower the barrier to re-entry. Implement a 'welcome back' fast-track onboarding process that skips standard checks [Source: Retention Handbook, Section 8E]"
    ]

def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_knowledge_base():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
    print(f"Building knowledge base at {db_path}...")
    
    client = chromadb.PersistentClient(path=db_path)
    
    # recreate collection
    try:
        client.delete_collection("retention_strategies")
    except:
        pass
        
    collection = client.create_collection("retention_strategies")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    docs = get_documents()
    
    all_chunks = []
    chunk_ids = []
    metadata = []
    
    chunk_id_counter = 0
    for doc in docs:
        chunks = chunk_text(doc, chunk_size=400, overlap=50)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_ids.append(f"chunk_{chunk_id_counter}")
            metadata.append({"source": "Retention Handbook"})
            chunk_id_counter += 1
            
    print(f"Generated {len(all_chunks)} chunks. Generating embeddings...")
    embeddings = model.encode(all_chunks).tolist()
    
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=metadata,
        ids=chunk_ids
    )
    print("Knowledge base successfully built!")

if __name__ == "__main__":
    build_knowledge_base()
