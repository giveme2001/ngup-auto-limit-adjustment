"""
자동한도조정 엔진 (Auto Limit Adjuster)
NGUP 가입설계 화면용

Author: Logic Developer Agent
Date: 2026-01-28
Version: 1.2

알고리즘: 비례 감액 (Proportional Reduction) + 하이브리드
- 설계사의 현재 설계를 최대한 존중
- 초과 시 각 특약의 기여 비율에 따라 균등하게 감액
- 자동 불가 시 사용자에게 수동 조정 알림
- 가입단위 정합성 보장 (올림)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import heapq
import time # 성능 측정용


# ============================================================
# 데이터 모델 정의
# ============================================================

# Priority enum 제거됨 (v1.2 비례 감액 방식에서는 우선순위 미사용)


@dataclass
class Rider:
    """
    특약 정보
    
    Attributes:
        rider_id: 특약 고유 ID
        name: 특약명
        current_amount: 현재 가입금액
        min_amount: 최소 가입금액
        max_amount: 최대 가입금액
        unit: 가입단위 (예: 1000만원)
        is_mandatory: 필수 특약 여부 (v1.2: 미사용, 호환성 유지)
        benefit_ids: 연결된 급부 ID 목록
        contribution_ratios: 급부별 기여 비율 (기본 1.0)
    """
    rider_id: str
    name: str
    current_amount: int
    min_amount: int
    max_amount: int
    unit: int
    is_mandatory: bool = False  # v1.2: 미사용, 호환성 유지
    is_locked: bool = False     # v1.3: 사용자 잠금 여부
    added_timestamp: float = 0.0 # v1.3: 추가된 순서 (LIFO/Greedy 정렬용)
    benefit_ids: List[str] = field(default_factory=list)
    contribution_ratios: Dict[str, float] = field(default_factory=dict)
    
    def get_contribution(self, benefit_id: str) -> int:
        """해당 급부에 대한 기여 금액 계산"""
        ratio = self.contribution_ratios.get(benefit_id, 1.0)
        return int(self.current_amount * ratio)
    
    def adjust_to_unit(self, amount: int) -> int:
        """가입단위에 맞게 금액 조정 (내림)"""
        return (amount // self.unit) * self.unit
    
    def can_reduce(self) -> bool:
        """감액 가능 여부"""
        return self.current_amount > self.min_amount
    
    def reducible_amount(self) -> int:
        """감액 가능한 최대 금액"""
        return self.current_amount - self.min_amount


@dataclass
class Benefit:
    """
    급부 정보
    
    Attributes:
        benefit_id: 급부 고유 ID
        name: 급부명
        cap: 급부 한도 (캡)
    """
    benefit_id: str
    name: str
    cap: int


@dataclass 
class AdjustResult:
    """
    조정 결과
    
    Attributes:
        success: 성공 여부
        adjusted_amounts: 조정된 금액 {rider_id: new_amount}
        changes: 변경 내역 {rider_id: (before, after, diff)}
        total_reduction: 총 감액 금액
        violations_fixed: 해결된 위반 수
        warnings: 경고 메시지 목록
        error: 에러 메시지 (실패 시)
    """
    success: bool
    adjusted_amounts: Dict[str, int] = field(default_factory=dict)
    changes: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    total_reduction: int = 0
    violations_fixed: int = 0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ============================================================
# 자동한도조정 엔진
# ============================================================

class AutoLimitAdjuster:
    """
    자동한도조정 엔진 v1.2
    
    비례 감액 (Proportional Reduction) 알고리즘으로
    설계사의 현재 설계를 최대한 존중하며 초과분을 균등 감액
    """
    
    MAX_ITERATIONS = 100  # 무한 루프 방지
    
    def __init__(self, riders: List[Rider], benefits: List[Benefit]):
        """
        Args:
            riders: 특약 목록
            benefits: 급부 목록
        """
        self.riders = {r.rider_id: r for r in riders}
        self.benefits = {b.benefit_id: b for b in benefits}
        
        # 급부별 연결된 특약 맵 구성
        self.benefit_riders: Dict[str, List[str]] = {}
        for rider in riders:
            for bid in rider.benefit_ids:
                if bid not in self.benefit_riders:
                    self.benefit_riders[bid] = []
                self.benefit_riders[bid].append(rider.rider_id)
        
        # 최적화용 캐시 (Incremental Update)
        self.current_benefit_totals: Dict[str, int] = {}
        self.rider_heap = [] # Priority Queue (Lazy Update): [(-score, rider_id, valid_token)]
        self.rider_valid_tokens: Dict[str, float] = {} # {rider_id: timestamp}
    
    def adjust(self, method: str = 'proportional') -> AdjustResult:
        """
        자동한도조정 실행
        
        Args:
            method: 'proportional' (기본), 'greedy', 'lifo'
        
        Returns:
            AdjustResult: 조정 결과
        """
        result = AdjustResult(success=False)
        
        # 원본 금액 저장
        original_amounts = {
            rid: r.current_amount for rid, r in self.riders.items()
        }
        
        # Step 2: 제약 만족할 때까지 반복
        
        # 최적화 초기화 (Greedy 모드일 때만)
        if method == 'greedy':
            self._initialize_optimization()
            
        for iteration in range(self.MAX_ITERATIONS):
            
            # Step 3: 전략별 감액 수행
            if method == 'greedy':
                # Global Greedy Strategy (v1.3 + v1.4 Optimization)
                # Incremental Update & Priority Queue & Batch Reduction
                
                # 위반 여부는 current_benefit_totals 캐시로 즉시 확인
                has_violation = False
                for bid, total in self.current_benefit_totals.items():
                    if total > self.benefits[bid].cap:
                        has_violation = True
                        break
                
                if not has_violation: break
                
                has_changed = self._step_global_greedy_reduction(result)
            else:
                # 기존 Single Violation Strategy (Proportional / LIFO)
                violation = self._find_violation()
                if violation is None: break  # 모든 제약 만족
                
                benefit_id, excess = violation
                result.violations_fixed += 1
                
                if method == 'lifo':
                    resolved, has_changed = self._resolve_violation_lifo(benefit_id, excess, result)
                else:
                    resolved, has_changed = self._resolve_violation_proportional(benefit_id, excess, result)
                
                if not resolved:
                     result.warnings.append(f"'{self.benefits[benefit_id].name}' 급부 해결 불가 가능성")

            # 무한루프 방지: 감액이 발생하지 않으면 조기 탈출
            if not has_changed:
                if method == 'greedy' and self._get_all_violations():
                     result.error = "ERR_STUCK: 더 이상 감액 가능한 특약이 없습니다."
                     break
                elif method != 'greedy' and self._find_violation():
                     break
        
        else:
            # 최대 반복 횟수 초과
            return AdjustResult(
                success=False,
                error="ERR_CIRCULAR: 순환 의존성으로 수렴 실패"
            )
        
        # Step 4: 결과 계산
        result.success = True
        for rid, rider in self.riders.items():
            result.adjusted_amounts[rid] = rider.current_amount
            
            original = original_amounts[rid]
            current = rider.current_amount
            diff = current - original
            
            if diff != 0:
                result.changes[rid] = (original, current, diff)
                result.total_reduction -= diff  # 감액은 음수이므로
        
        return result
    
    def _find_violation(self) -> Optional[Tuple[str, int]]:
        """첫 번째 제약 위반 탐지 (기존 방식)"""
        for benefit_id, benefit in self.benefits.items():
            total = self._calculate_benefit_total(benefit_id)
            
            if total > benefit.cap:
                excess = total - benefit.cap
                return (benefit_id, excess)
        return None

    def _get_all_violations(self) -> Dict[str, int]:
        """모든 제약 위반 탐지 (Global Greedy용)"""
        violations = {}
        for benefit_id, benefit in self.benefits.items():
            total = self._calculate_benefit_total(benefit_id)
            if total > benefit.cap:
                violations[benefit_id] = total - benefit.cap
        return violations
    
    def _calculate_benefit_total(self, benefit_id: str) -> int:
        """특정 급부의 총 기여 금액 계산 (캐시 활용 가능 시 증분값 아님 - 초기화용)"""
        # 만약 최적화 모드이고 캐시가 있다면 캐시 반환? 
        # 아니오, 초기화나 검증용이므로 직접 계산.
        total = 0
        rider_ids = self.benefit_riders.get(benefit_id, [])
        
        for rid in rider_ids:
            rider = self.riders[rid]
            total += rider.get_contribution(benefit_id)
        
        return total
        
    def _get_reducible_riders(self, benefit_id: str) -> List[Rider]:
        """해당 급부에 연결된 감액 가능한 특약 목록 반환"""
        rider_ids = self.benefit_riders.get(benefit_id, [])
        riders = [self.riders[rid] for rid in rider_ids]
        return [r for r in riders if r.can_reduce()]
    
    def _apply_reduction(self, rider: Rider, target_reduce: int, result: AdjustResult) -> int:
        """
        공통 감액 로직: 가입단위 고려하여 실제 감액량 계산 및 적용
        Returns: actual_reduced_amount
        """
        reducible = rider.reducible_amount()
        reduce_amount = min(target_reduce, reducible)
        
        if rider.unit > 0:
            units_needed = math.ceil(reduce_amount / rider.unit)
            actual_reduce = units_needed * rider.unit
            actual_reduce = min(actual_reduce, reducible)
        else:
            actual_reduce = reduce_amount
            
        if actual_reduce > 0:
            rider.current_amount -= actual_reduce
            if rider.current_amount == rider.min_amount:
                result.warnings.append(f"'{rider.name}' 특약이 최소 가입금액에 도달")
            return actual_reduce
            
        return 0

    def _apply_reduction_strategy(
        self, riders: List[Rider], excess: int, result: AdjustResult
    ) -> Tuple[bool, bool]:
        """순차적 감액 전략 공통 로직 (Greedy/LIFO용)"""
        remaining = excess
        has_changed = False
        
        for rider in riders:
            if remaining <= 0: break
            
            actual_reduce = self._apply_reduction(rider, remaining, result)
            
            if actual_reduce > 0:
                remaining -= actual_reduce
                has_changed = True
                
        return (remaining <= 0, has_changed)

    def _resolve_violation_proportional(
        self, benefit_id: str, excess: int, result: AdjustResult
    ) -> Tuple[bool, bool]:
        """비례 감액 전략"""
        reducible_riders = self._get_reducible_riders(benefit_id)
        if not reducible_riders: return (False, False)
        
        # 기여금 총합 계산
        contributions = {r.rider_id: r.get_contribution(benefit_id) for r in reducible_riders}
        total_contribution = sum(contributions.values())
        
        if total_contribution == 0: return (False, False)
        
        remaining = excess
        has_changed = False
        
        # 1차: 비례 감액
        for rider in reducible_riders:
            if remaining <= 0: break
            
            ratio = contributions[rider.rider_id] / total_contribution
            target_reduce = int(excess * ratio)
            
            actual_reduce = self._apply_reduction(rider, target_reduce, result)
            if actual_reduce > 0:
                remaining -= actual_reduce
                has_changed = True
        
        # 2차: 잔여분 처리 (순차적으로 적용)
        if remaining > 0:
            sub_resolved, sub_changed = self._apply_reduction_strategy(reducible_riders, remaining, result)
            if sub_changed: has_changed = True
            if sub_resolved: remaining = 0
            
        return (remaining <= 0, has_changed)

    def _calculate_greedy_score(self, rider: Rider, violations: Dict[str, int]) -> int:
        """
        Greedy 감액 우선순위 점수(S) 계산
        S = (V * 10^10) + (E * 10^5) + A
        
        V: Violation Count (현재 위반 중인 급부 중 이 특약이 포함된 개수)
        E: Efficiency (단위 감액 시 줄어드는 총 초과액 / 단위) -> 단위당 효율
           * 사용자 정의: "특약 가입금액을 1단위 줄였을 때, 모든 위반 급부에서 줄어드는 초과액의 총합"
             -> 즉, Unit으로 나누지 않고, "1 Unit Reduction Effect" 그 자체를 의미하는 것으로 해석됨.
             -> 하지만 "감액 효율 지수" 정의에 "/ Unit"이 있었으므로, 여기서는 
                "1 Unit을 줄였을 때(Action) 얻는 이득(Gain)" = Sum(min(ratio*unit, excess))
                이 값 자체가 크면 좋음.
        A: Current Amount
        """
        if rider.is_locked: return -1
        if not rider.can_reduce(): return -1
        
        # 1. V (Violation Count)
        violation_count = 0
        current_violating_benefits = []
        for bid in rider.benefit_ids:
            if bid in violations:
                violation_count += 1
                current_violating_benefits.append(bid)
        
        if violation_count == 0: return -1
        
        # 2. E (Efficiency) - 1 Unit 감액 시 효과
        # 특약을 1 Unit(또는 단위가 0이면 1) 줄였을 때, 위반 급부들의 Excess가 얼마나 줄어드는가?
        unit = rider.unit if rider.unit > 0 else 1
        total_reduction_effect = 0
        
        for bid in current_violating_benefits:
            ratio = rider.contribution_ratios.get(bid, 1.0)
            reduction_amount_for_benefit = int(unit * ratio)
            # 해당 급부의 남은 초과분까지만 효과 인정 (이미 해결된 거 더 깎아봤자 의미 없음 -> 근데 여기선 그냥 기여분만큼 인정)
            # 하지만 정확한 "효율"은 "유효한 감액분"이므로 min 처리가 맞음
            effect = min(reduction_amount_for_benefit, violations[bid])
            total_reduction_effect += effect
            
        # E calculation: 1 Unit을 깎았을 때의 효과 총합.
        # 만약 Unit이 제각각이라면 Unit으로 나누어 "화폐단위당 효율"을 봐야 할 수도 있으나,
        # 기획서 상 "특약 가입금액을 1단위(unit) 줄였을 때" 라고 명시됨. -> Action 기준.
        # 또한 가입단위가 큰 놈을 건드리는 게(큰 덩어리) 시원하게 빠지므로 E가 클 가능성 높음.
        efficiency = total_reduction_effect
        
        # 3. A (Current Amount)
        current_amount = rider.current_amount
        
        # Score S
        score = (violation_count * (10**10)) + (efficiency * (10**5)) + current_amount
        return score

    def _initialize_optimization(self):
        """Greedy 최적화를 위한 초기화 (Totals & Heap Building)"""
        # 1. Benefit Totals 초기화
        self.current_benefit_totals = {}
        for bid in self.benefits:
            self.current_benefit_totals[bid] = self._calculate_benefit_total(bid)
            
        # 2. Heap 초기화
        # 초기 위반 상태 파악
        violations = {
            bid: self.current_benefit_totals[bid] - self.benefits[bid].cap
            for bid, total in self.current_benefit_totals.items()
            if total > self.benefits[bid].cap
        }
        
        self.rider_heap = []
        self.rider_valid_tokens = {}
        
        timestamp = time.time()
        for rid, rider in self.riders.items():
            self.rider_valid_tokens[rid] = timestamp
            score = self._calculate_greedy_score(rider, violations)
            if score > 0: # 감액 가능한 후보만
                heapq.heappush(self.rider_heap, (-score, rid, timestamp))

    def _update_rider_in_heap(self, rider: Rider, violations: Dict[str, int]):
        """변경된 특약의 점수를 재계산하여 Heap에 Push (Lazy Update)"""
        timestamp = time.time()
        self.rider_valid_tokens[rider.rider_id] = timestamp
        score = self._calculate_greedy_score(rider, violations)
        # 점수가 유효할 때만 push (0 이하면 제외)
        if score > 0:
            heapq.heappush(self.rider_heap, (-score, rider.rider_id, timestamp))

    def _step_global_greedy_reduction(self, result: AdjustResult) -> bool:
        """Global Greedy Optimized: Heap에서 최적 특약 추출 후 Batch 감액"""
        
        # 1. 현재 위반 상태 추출 (Incremental Cache 사용)
        violations = {
            bid: self.current_benefit_totals[bid] - self.benefits[bid].cap
            for bid, total in self.current_benefit_totals.items()
            if total > self.benefits[bid].cap
        }
        
        if not violations: return False
        
        # 2. Heap에서 최적 특약 Pop (Lazy Update Check)
        best_rider: Optional[Rider] = None
        
        while self.rider_heap:
            neg_score, rid, token = heapq.heappop(self.rider_heap)
            
            # 유효성 검사 (Stale Token Check)
            if token != self.rider_valid_tokens.get(rid):
                continue
            
            # 현재 시점의 점수와 비교? 
            # -> 엄밀히는 위반 상태(Violations)가 변했으므로 점수도 변했을 수 있음.
            # -> 하지만 모든 특약 점수를 매번 갱신하면 Heap의 장점이 사라짐.
            # -> "위반 급부의 해소"가 발생했을 때만 관련 특약들을 갱신해야 함.
            # -> 여기서는 "Pop 한 시점"에 점수를 다시 계산해보고, 
            #    만약 힙 상의 점수와 크게 다르다면(위반 상황 변경됨) 다시 Push 하고 다음 거 뽑기? 
            #    (Heuristic: 그냥 쓴다. V가 가장 중요하니까.)
            
            best_rider = self.riders[rid]
            break
            
        if best_rider is None:
            # 힙이 비었거나 유효한 놈이 없음 -> 다시 스캔 필요 (혹은 종료)
            # 안전장치: 전체 재스캔 한번 시도
            self.rider_valid_tokens = {} # Reset
            timestamp = time.time()
            count = 0
            for rid, rider in self.riders.items():
                self.rider_valid_tokens[rid] = timestamp
                s = self._calculate_greedy_score(rider, violations)
                if s > 0:
                    heapq.heappush(self.rider_heap, (-s, rid, timestamp))
                    count += 1
            
            if count == 0: return False # 진짜 없음
            
            # 재시도
            neg_score, rid, token = heapq.heappop(self.rider_heap)
            best_rider = self.riders[rid]
            
        # 3. Batch 감액량 계산 (Batch Reduction)
        # 해당 특약이 기여하는 위반 급부들 중, 가장 '조금' 깎아도 해결되는 양? 아니면 '많이' 깎아야 하는 양?
        # -> 가장 위반이 심한 것을 해결하려면 많이 깎아야 함.
        # -> 하지만 다른 급부의 초과분이 작다면? 
        # -> 안전하게: 1 Unit 보다는 크되, 과도하지 않게.
        # -> 정책: "위반액 / 기여율" 만큼 한 번에 깎음.
        # -> 여러 위반 급부가 연결되어 있다면? 가장 큰 위반액(Critical Path)을 기준으로 삼는다.
        
        max_needed_reduction = 0
        for bid in best_rider.benefit_ids:
            if bid in violations:
                ratio = best_rider.contribution_ratios.get(bid, 1.0)
                if ratio > 0:
                    needed = violations[bid] / ratio
                    if needed > max_needed_reduction:
                        max_needed_reduction = needed
        
        if max_needed_reduction == 0: return False # Should not happen
        
        # Unit 단위 보정 (올림)
        unit = best_rider.unit if best_rider.unit > 0 else 1
        units_needed = math.ceil(max_needed_reduction / unit)
        target_reduce = units_needed * unit
        
        # 최소값 보호
        reducible = best_rider.current_amount - best_rider.min_amount
        actual_reduce = min(target_reduce, reducible)
        
        if actual_reduce <= 0:
            return False
            
        # 4. 감액 실행 및 증분 업데이트 (Incremental Update)
        best_rider.current_amount -= actual_reduce
        
        # 전파 (Propagate): 연결된 급부들의 Total 캐시 갱신
        affected_benefits = []
        for bid in best_rider.benefit_ids:
            ratio = best_rider.contribution_ratios.get(bid, 1.0)
            reduction_effect = int(actual_reduce * ratio)
            if bid in self.current_benefit_totals:
                self.current_benefit_totals[bid] -= reduction_effect
                affected_benefits.append(bid)
        
        # 5. Heap 갱신 (관련 특약들만 점수 재계산 필요...하지만 너무 많을 수 있음)
        # -> 일단 본인은 갱신해서 다시 넣음.
        new_violations = {
            bid: self.current_benefit_totals[bid] - self.benefits[bid].cap
            for bid, total in self.current_benefit_totals.items()
            if total > self.benefits[bid].cap
        }
        self._update_rider_in_heap(best_rider, new_violations)
        
        # 심화 최적화: 사실 affected_benefits에 연결된 *다른 모든 특약*들의 점수도 변했을 수 있음 (V 값이 변하므로).
        # 하지만 이걸 다 갱신하면 O(N^2) 됨. 
        # -> Global Loop의 특성상, 이번에 안 뽑혀도 다음 루프에 뽑힐 것임. 
        # -> 일단 본인만 갱신하는 Lazy 전략 유지.
        
        return True

    def _resolve_violation_lifo(
        self, benefit_id: str, excess: int, result: AdjustResult
    ) -> Tuple[bool, bool]:
        """LIFO 전략 (나중에 추가된 순서)"""
        reducible_riders = self._get_reducible_riders(benefit_id)
        if not reducible_riders: return (False, False)
        
        # 역순 정렬 (가정: 리스트는 추가된 순서)
        reducible_riders.reverse()
        return self._apply_reduction_strategy(reducible_riders, excess, result)

    def get_benefit_status(self) -> Dict[str, Dict]:
        """
        모든 급부의 현재 상태 조회
        
        Returns:
            {benefit_id: {total, cap, usage_pct, is_over}}
        """
        status = {}
        
        for bid, benefit in self.benefits.items():
            total = self._calculate_benefit_total(bid)
            usage_pct = (total / benefit.cap * 100) if benefit.cap > 0 else 0
            
            status[bid] = {
                "name": benefit.name,
                "total": total,
                "cap": benefit.cap,
                "usage_pct": round(usage_pct, 1),
                "is_over": total > benefit.cap,
                "excess": max(0, total - benefit.cap)
            }
        
        return status


# ============================================================
# 테스트
# ============================================================

def create_test_data():
    riders = [
        # Case 2: Multi-Benefit Greedy Test
        # R001: B001, B002 동시 위반 (V=2) -> 우선 감액 대상이어야 함
        Rider("R001", "다중위반특약", 50000000, 10000000, 100000000, 10000000, benefit_ids=["B001", "B002"], contribution_ratios={"B001": 1.0, "B002": 1.0}),
        # R002: B001만 위반 (V=1)
        Rider("R002", "단일위반특약", 50000000, 10000000, 50000000, 10000000, benefit_ids=["B001"], contribution_ratios={"B001": 1.0}),
    ] 
    
    benefits = [
        Benefit("B001", "급부1", 40000000), # R001(5천)+R002(5천)=1억 -> 6천 초과
        Benefit("B002", "급부2", 40000000), # R001(5천)=5천 -> 1천 초과
    ]
    return riders, benefits

def run_tests():
    """테스트 케이스 실행"""
    print("=" * 60)
    print("자동한도조정 엔진 테스트 (v1.2 Multi-Algorithm)")
    print("=" * 60)
    
    methods = [
        ("proportional", "Proportional (비례)"),
        ("greedy", "Greedy (금액순)"),
        ("lifo", "LIFO (역순)")
    ]
    
    for method_code, method_name in methods:
        print(f"\n[테스트] {method_name}")
        print("-" * 40)
        
        riders, benefits = create_test_data()
        engine = AutoLimitAdjuster(riders, benefits)
        
        # 조정 전 상태 출력 (첫 번째만)
        if method_code == "proportional":
            print("조정 전 상태:")
            for r in riders: 
                lock_status = "[LOCKED]" if r.is_locked else ""
                print(f"  - {r.name}{lock_status}: {r.current_amount:,}원")
        
        result = engine.adjust(method=method_code)
        
        if result.success:
            print(f"결과: 성공 (총 감액 {result.total_reduction:,}원)")
            for rid, (before, after, diff) in result.changes.items():
                rider = engine.riders[rid]
                print(f"  {rider.name}: {before:,} → {after:,} ({diff:,})")
        else:
            print(f"결과: 실패 ({result.error})")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_tests()


